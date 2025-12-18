    def reduce_attention_scores(self, attn_scores):
        batch_size, num_heads, seq_len_q, seq_len_k = attn_scores.shape
        local_scores_reshaped = attn_scores.reshape(batch_size, num_heads, seq_len_q * seq_len_k)
        reduced_scores = reduce_from_tensor_model_parallel_region(local_scores_reshaped)               
        return reduced_scores.reshape(batch_size, -1, seq_len_q, seq_len_k)

    import nvtx
    @nvtx.annotate(message="recompute_attention", color="red")
    def recompute_attention_scores(self, query, key, attention_mask, scaling):
        # query: [s, b, n, h] -> [b, n, s, h]
        query = query.permute(1, 2, 0, 3)
        # key: [s, b, h] -> [b, s, h]
        key = key.permute(1, 0, 2)

        attn_scores = torch.einsum("bhsc,btc->bhst", query, key) * scaling
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill_(attention_mask, -1e9)

        attn_scores = torch.softmax(attn_scores, dim=-1)
        return self.reduce_attention_scores(attn_scores)



indexer_q_detached = self.indexer_query.detach().requires_grad_(True)
            indexer_k_detached = self.indexer_key.detach().requires_grad_(True)
            indexer_w_detached = self.indexer_weights.detach().requires_grad_(True)
                
            topk_indices_list = []
            chunk_size = self.lightning_indexer.index_chunk_size
            total_chunks = (query_mqa.size(0) + chunk_size - 1) // chunk_size

            indexer_loss = 0.0

            for i in range(total_chunks):
                start_idx = i * chunk_size  # 0 MB
                end_idx = min((i + 1) * chunk_size, query_mqa.size(0))
                causal_mask = torch.triu(torch.ones(chunk_size, query_mqa.size(0), device="cuda"), diagonal=start_idx).bool()
                
                chunk_indexer_scores = self.lightning_indexer.indexer(
                    indexer_q_detached[:, start_idx:end_idx],   # [B, chunk_size, H, D]
                    indexer_k_detached,                           # [B, Seq_k, D]
                    indexer_w_detached[start_idx:end_idx]     # [chunk_size, B, H]
                )

                chunk_indexer_scores = chunk_indexer_scores.masked_fill_(causal_mask, float('-inf'))
                topk_indices = chunk_indexer_scores.topk(self.lightning_indexer.index_topk, dim=-1)[1]
                topk_indices_list.append(topk_indices)
                index_mask = torch.full(
                    chunk_indexer_scores.shape, 
                    True, 
                    device=chunk_indexer_scores.device
                ).scatter_(-1, topk_indices, False)
                index_mask = torch.logical_or(index_mask, causal_mask).unsqueeze(1)
                    
                with torch.no_grad():
                    chunk_attn_scores = self.fused_recompute_attention(
                        query_mqa[start_idx:end_idx],        # [chunk_size, B, H, D]
                        kv_mqa,                              # [Seq_k, B, H, D]
                        index_mask,
                        self.softmax_scale
                    )

                loss = self.lightning_indexer.fused_index_loss(
                    chunk_indexer_scores, 
                    chunk_attn_scores, 
                    index_mask.squeeze(0)
                )

                loss.backward()
                indexer_loss = indexer_loss + loss

            self.indexer_query.backward(indexer_q_detached.grad)
            self.indexer_key.backward(indexer_k_detached.grad)
            self.indexer_weights.backward(indexer_w_detached.grad)
            self.indexer_query = None
            self.indexer_key = None
            self.indexer_weights = None


    def compute_index_loss(
        self,
        index_score,
        attention_scores,
        index_mask=None,
    ):
        """
        Computes the KL loss between the index score and the attention scores.

        Args:
            index_score: The index score tensor of shape (bsz, seq, seq).
            attention_scores: The attention scores tensor of shape (bsz, num_head, seq, seq).
            index_mask: An optional mask tensor of shape (bsz, seq, seq) indicating which positions are valid.

        Returns:
            A scalar tensor representing the KL loss.
        """

        eps = 1e-10

        assert attention_scores.dim() == 4              # (bsz, head, seq_len, seq_len)
        attention_scores = attention_scores.sum(1)

        if index_mask is not None:
            index_score = index_score.masked_fill(index_mask, -1e9)

        index_score = torch.softmax(index_score, dim=-1) + eps
        attn_dist = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

        kl_loss = F.kl_div(index_score.log(), attn_dist + eps, reduction="batchmean")
        return kl_loss
