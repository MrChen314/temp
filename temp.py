    attn_sum_agg = attn_sum.sum(dim=0)
    attn_dist = attn_sum_agg / (attn_sum_agg.sum(dim=-1, keepdim=True) + eps)
