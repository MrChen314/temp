        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda"), diagonal=1).bool()

        causal_index_score = index_score.masked_fill(causal_mask, float('-inf'))
        topk_indices = causal_index_score.topk(self.index_topk, dim=-1)[1]


        index_mask = torch.full(
            causal_index_score.shape, 
            True, 
            device=index_score.device
        ).scatter_(-1, topk_indices, False)
        index_mask = torch.logical_or(index_mask, causal_mask)
        index_mask = index_mask.unsqueeze(1)
