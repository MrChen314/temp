    def calc_diff(a, b):
        abs_diff = torch.abs(a - b)
        max_diff = abs_diff.max().item()
        rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100
        return max_diff, rel_diff
    
    max_diff, rel_diff = calc_diff(ref_attn_dist, tl_attn_dist.to(ref_attn_dist.dtype))
    passed = rel_diff < 1e-3  # relative error < 0.001%
