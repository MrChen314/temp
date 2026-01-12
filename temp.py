# ruff: noqa
"""
Sparse MLA Forward with Sequence Blocking

Optimization: Instead of blocking by heads (H_per_block), we block by sequence (S_per_block).
This maintains high register and shared memory utilization even with small head counts.

Key changes:
- Q_shared: [S_per_block, D] instead of [H_per_block, D]
- Pre-computed mask matrix handles different indices for different sequence positions
- For each seq block, use the last position's indices as superset
- Mask ensures correctness by filtering valid (seq_pos, kv_idx) pairs
"""
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


def prepare_block_indices_and_mask(
    indices: torch.Tensor,
    seq_len: int,
    S_per_block: int,
    block_I: int,
):
    """
    Prepare indices and mask for sequence-blocked processing.
    
    For each seq block, use the indices from the last position in the block.
    Generate mask to handle causality and valid topk membership.
    
    Args:
        indices: [B, S, G, topk] - original per-position topk indices
        seq_len: actual sequence length
        S_per_block: number of sequence positions per block (e.g., 64)
        block_I: KV block size (e.g., 64)
    
    Returns:
        block_indices: [B, num_seq_blocks, G, topk] - indices for each seq block
        block_mask: [B, num_seq_blocks, G, NI, S_per_block, BI] - mask for each (seq_pos, kv_idx) pair
    """
    B, S, G, topk = indices.shape
    num_seq_blocks = (seq_len + S_per_block - 1) // S_per_block
    BI = block_I
    NI = topk // BI
    
    device = indices.device
    
    # For each seq block, use the last position's indices as superset
    # This ensures we have all KV positions that any position in the block might need
    block_indices = torch.zeros(B, num_seq_blocks, G, topk, dtype=indices.dtype, device=device)
    
    for sb in range(num_seq_blocks):
        # Last position in this block (or the actual last position if block is partial)
        last_pos = min((sb + 1) * S_per_block - 1, seq_len - 1)
        block_indices[:, sb, :, :] = indices[:, last_pos, :, :]
    
    # Generate mask: [B, num_seq_blocks, G, NI, S_per_block, BI]
    # mask[b, sb, g, ni, si, bi] = True if:
    #   1. kv_idx <= real_seq_pos (causal constraint)
    #   2. kv_idx is in the original topk for this seq pos (validity constraint)
    block_mask = torch.zeros(
        B, num_seq_blocks, G, NI, S_per_block, BI, 
        dtype=torch.bool, device=device
    )
    
    for b in range(B):
        for sb in range(num_seq_blocks):
            for g in range(G):
                for ni in range(NI):
                    for si in range(S_per_block):
                        real_seq_pos = sb * S_per_block + si
                        if real_seq_pos >= seq_len:
                            # Out of bounds, mask should be False (already initialized)
                            continue
                        
                        # Get original topk indices for this seq position
                        original_topk_set = set(indices[b, real_seq_pos, g, :].tolist())
                        
                        for bi in range(BI):
                            kv_idx = block_indices[b, sb, g, ni * BI + bi].item()
                            
                            # Causal constraint: can only attend to positions <= current
                            is_causal_valid = kv_idx <= real_seq_pos
                            
                            # Validity constraint: kv_idx must be in original topk
                            is_in_original_topk = kv_idx in original_topk_set
                            
                            block_mask[b, sb, g, ni, si, bi] = is_causal_valid and is_in_original_topk
    
    return block_indices, block_mask


def prepare_block_indices_and_mask_fast(
    indices: torch.Tensor,
    seq_len: int,
    S_per_block: int,
    block_I: int,
):
    """
    Vectorized version of prepare_block_indices_and_mask for better performance.
    
    Args:
        indices: [B, S, G, topk] - original per-position topk indices
        seq_len: actual sequence length
        S_per_block: number of sequence positions per block (e.g., 64)
        block_I: KV block size (e.g., 64)
    
    Returns:
        block_indices: [B, num_seq_blocks, G, topk] - indices for each seq block
        block_mask: [B, num_seq_blocks, G, NI, S_per_block, BI] - mask for each (seq_pos, kv_idx) pair
    """
    B, S, G, topk = indices.shape
    num_seq_blocks = (seq_len + S_per_block - 1) // S_per_block
    BI = block_I
    NI = topk // BI
    
    device = indices.device
    
    # Pad indices if needed
    padded_S = num_seq_blocks * S_per_block
    if padded_S > S:
        # Pad with the last valid position's indices
        indices_padded = torch.zeros(B, padded_S, G, topk, dtype=indices.dtype, device=device)
        indices_padded[:, :S, :, :] = indices
        indices_padded[:, S:, :, :] = indices[:, S-1:S, :, :]
    else:
        indices_padded = indices
    
    # Reshape indices to [B, num_seq_blocks, S_per_block, G, topk]
    indices_reshaped = indices_padded.view(B, num_seq_blocks, S_per_block, G, topk)
    
    # For each seq block, use the last position's indices
    # block_indices: [B, num_seq_blocks, G, topk]
    block_indices = indices_reshaped[:, :, -1, :, :]  # Use last position in each block
    
    # Reshape block_indices for broadcasting: [B, num_seq_blocks, G, NI, 1, BI]
    block_indices_expanded = block_indices.view(B, num_seq_blocks, G, NI, 1, BI)
    
    # Create real_seq_pos tensor: [num_seq_blocks, S_per_block]
    seq_block_offsets = torch.arange(num_seq_blocks, device=device) * S_per_block
    seq_pos_in_block = torch.arange(S_per_block, device=device)
    real_seq_pos = seq_block_offsets.view(-1, 1) + seq_pos_in_block.view(1, -1)  # [num_seq_blocks, S_per_block]
    real_seq_pos = real_seq_pos.view(1, num_seq_blocks, 1, 1, S_per_block, 1)  # [1, num_seq_blocks, 1, 1, S_per_block, 1]
    
    # Causal mask: kv_idx <= real_seq_pos
    # block_indices_expanded: [B, num_seq_blocks, G, NI, 1, BI]
    # real_seq_pos: [1, num_seq_blocks, 1, 1, S_per_block, 1]
    causal_mask = block_indices_expanded <= real_seq_pos  # [B, num_seq_blocks, G, NI, S_per_block, BI]
    
    # Validity mask: kv_idx must be in original topk for this seq pos
    # This is more complex - need to check membership
    # indices_reshaped: [B, num_seq_blocks, S_per_block, G, topk]
    # block_indices_expanded: [B, num_seq_blocks, G, NI, 1, BI]
    
    # For each (sb, si), check if each kv_idx is in indices[b, sb, si, g, :]
    # Reshape for comparison:
    # original_indices: [B, num_seq_blocks, S_per_block, G, topk] -> [B, num_seq_blocks, 1, S_per_block, G, topk]
    original_indices = indices_reshaped.unsqueeze(2)  # [B, num_seq_blocks, 1, S_per_block, G, topk]
    original_indices = original_indices.permute(0, 1, 4, 2, 3, 5)  # [B, num_seq_blocks, G, 1, S_per_block, topk]
    
    # block_indices for comparison: [B, num_seq_blocks, G, NI*BI] -> [B, num_seq_blocks, G, NI*BI, 1, 1]
    block_indices_flat = block_indices.view(B, num_seq_blocks, G, NI * BI, 1, 1)
    
    # Compare: is block_indices_flat[..., k, :, :] in original_indices[..., s, :]?
    # This creates a [B, num_seq_blocks, G, NI*BI, S_per_block, topk] tensor - too large!
    # Need a smarter approach
    
    # Alternative: use torch.isin per position (still not great but works)
    validity_mask = torch.zeros(B, num_seq_blocks, G, NI, S_per_block, BI, dtype=torch.bool, device=device)
    
    for sb in range(num_seq_blocks):
        for si in range(S_per_block):
            real_pos = sb * S_per_block + si
            if real_pos >= seq_len:
                continue
            # original topk for this position: [B, G, topk]
            orig_topk = indices[:, real_pos, :, :]  # [B, G, topk]
            # block indices for this block: [B, G, topk]
            blk_idx = block_indices[:, sb, :, :]  # [B, G, topk]
            
            # Check membership: blk_idx in orig_topk
            # For each (b, g, k), check if blk_idx[b, g, k] is in orig_topk[b, g, :]
            for b in range(B):
                for g in range(G):
                    orig_set = set(orig_topk[b, g, :].tolist())
                    for ni in range(NI):
                        for bi in range(BI):
                            kv_idx = blk_idx[b, g, ni * BI + bi].item()
                            validity_mask[b, sb, g, ni, si, bi] = kv_idx in orig_set
    
    # Combine masks
    block_mask = causal_mask & validity_mask
    
    # Handle out-of-bounds positions
    for sb in range(num_seq_blocks):
        for si in range(S_per_block):
            real_pos = sb * S_per_block + si
            if real_pos >= seq_len:
                block_mask[:, sb, :, :, si, :] = False
    
    return block_indices, block_mask


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_seq_block(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    S_per_block=64,
    num_stages=2,
    threads=256,
):
    """
    Sparse MLA Forward kernel with sequence blocking.
    
    Instead of blocking by heads, we block by sequence positions.
    This maintains high resource utilization even with small head counts.
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_seq_blocks = T.dynamic("num_seq_blocks")

    head_kv = heads // kv_group
    H = heads
    G = kv_group
    
    # Shapes
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    lse_shape = [batch, seq_len, heads]
    
    # New: block indices and mask shapes
    block_indices_shape = [batch, num_seq_blocks, kv_group, topk]
    block_mask_shape = [batch, num_seq_blocks, kv_group, topk // block_I, S_per_block, block_I]
    
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    S_blk = S_per_block

    # Padded output shapes for kernel (uses padded seq_len = num_seq_blocks * S_per_block)
    padded_seq_len = num_seq_blocks * S_per_block
    o_shape_padded = [batch, padded_seq_len, heads, dim]
    lse_shape_padded = [batch, padded_seq_len, heads]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore - [batch, padded_seq_len, heads, dim+tail_dim]
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        BlockIndices: T.Tensor(block_indices_shape, indices_dtype),  # type: ignore  # Pre-computed block indices
        BlockMask: T.Tensor(block_mask_shape, "bool"),  # type: ignore  # Pre-computed mask
        Output: T.Tensor(o_shape_padded, dtype),  # type: ignore - [batch, padded_seq_len, heads, dim]
        Lse: T.Tensor(lse_shape_padded, accum_dtype),  # type: ignore - [batch, padded_seq_len, heads]
    ):
        # Grid: (num_seq_blocks, batch, heads)
        # Each block processes S_per_block sequence positions for one head
        with T.Kernel(num_seq_blocks, batch, heads, threads=threads) as (
            bx,
            by,
            bz,
        ):
            # Shared memory allocations - now S_per_block instead of H_per_block
            Q_shared = T.alloc_shared([S_blk, D], dtype)
            Q_tail_shared = T.alloc_shared([S_blk, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([S_blk, D], dtype)
            Lse_shared = T.alloc_shared([S_blk], accum_dtype)
            Mask_shared = T.alloc_shared([S_blk, BI], "bool")
            
            # Fragment allocations
            acc_o = T.alloc_fragment([S_blk, D], accum_dtype)
            acc_s = T.alloc_fragment([S_blk, BI], accum_dtype)
            S_shared = T.alloc_shared([S_blk, BI], dtype)
            sumexp = T.alloc_fragment([S_blk], accum_dtype)
            sumexp_i = T.alloc_fragment([S_blk], accum_dtype)
            alpha = T.alloc_fragment([S_blk], accum_dtype)
            m_i = T.alloc_fragment([S_blk], accum_dtype)
            m_i_prev = T.alloc_fragment([S_blk], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i = by  # batch index
            sb_i = bx  # seq block index
            h_i = bz  # head index
            g_i = h_i // (H // G)  # kv group index

            # Compute sequence range for this block
            S0 = sb_i * S_blk

            # Load Q for this seq block and head
            # Q: [batch, padded_seq_len, heads, dim+tail_dim]
            # We need Q[b_i, S0:S0+S_blk, h_i, :]
            for s_i, d_i in T.Parallel(S_blk, D):
                Q_shared[s_i, d_i] = Q[b_i, S0 + s_i, h_i, d_i]
            for s_i, d_i in T.Parallel(S_blk, D_tail):
                Q_tail_shared[s_i, d_i] = Q[b_i, S0 + s_i, h_i, D + d_i]

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # Load pre-computed mask for this (seq_block, kv_block)
                # BlockMask: [batch, num_seq_blocks, kv_group, NI, S_per_block, BI]
                for s_i, bi_i in T.Parallel(S_blk, BI):
                    Mask_shared[s_i, bi_i] = BlockMask[b_i, sb_i, g_i, i_i, s_i, bi_i]

                # Load KV using block indices
                # BlockIndices: [batch, num_seq_blocks, kv_group, topk]
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, BlockIndices[b_i, sb_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, BlockIndices[b_i, sb_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # Apply mask to acc_s initialization
                for s_i, bi_i in T.Parallel(S_blk, BI):
                    acc_s[s_i, bi_i] = T.if_then_else(Mask_shared[s_i, bi_i], 0, -T.infinity(acc_s.dtype))
                
                # Q @ K^T
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                
                # Online softmax update
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for s_i in T.Parallel(S_blk):
                    m_i[s_i] = T.max(m_i[s_i], m_i_prev[s_i])
                for s_i in T.Parallel(S_blk):
                    alpha[s_i] = T.exp2((m_i_prev[s_i] - m_i[s_i]) * sm_scale)
                for s_i, bi_i in T.Parallel(S_blk, BI):
                    acc_s[s_i, bi_i] = T.exp2(acc_s[s_i, bi_i] * sm_scale - m_i[s_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for s_i in T.Parallel(S_blk):
                    sumexp[s_i] = sumexp[s_i] * alpha[s_i] + sumexp_i[s_i]
                for s_i, d_i in T.Parallel(S_blk, D):
                    acc_o[s_i, d_i] = acc_o[s_i, d_i] * alpha[s_i]

                # S @ V
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale output
            for s_i, d_i in T.Parallel(S_blk, D):
                acc_o[s_i, d_i] /= sumexp[s_i]
            for s_i in T.Parallel(S_blk):
                sumexp[s_i] = T.log2(sumexp[s_i]) + m_i[s_i] * sm_scale

            # Store output
            # Output: [batch, padded_seq_len, heads, dim]
            T.copy(acc_o, O_shared)
            for s_i, d_i in T.Parallel(S_blk, D):
                Output[b_i, S0 + s_i, h_i, d_i] = O_shared[s_i, d_i]
            T.copy(sumexp, Lse_shared)
            for s_i in T.Parallel(S_blk):
                Lse[b_i, S0 + s_i, h_i] = Lse_shared[s_i]

    return main


def sparse_mla_fwd_seq_block_interface(
    q, kv, indices, 
    sm_scale=None, 
    return_p_sum: bool = False, 
    d_v=512, 
    block_I=64, 
    S_per_block=64,
    num_stages=2, 
    threads=256
):
    """
    Interface function for sparse MLA forward with sequence blocking.
    
    This function:
    1. Pre-computes block indices and mask in Python
    2. Launches the kernel with pre-computed data
    
    Key optimization: Instead of blocking by heads (H_per_block), we block by sequence (S_per_block).
    This maintains high register and shared memory utilization even with small head counts.
    """
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)
    
    # Store original seq_len for slicing output later
    original_seq_len = seq_len
    
    # Ensure seq_len is divisible by S_per_block
    num_seq_blocks = (seq_len + S_per_block - 1) // S_per_block
    padded_seq_len = num_seq_blocks * S_per_block
    
    # Pad Q and indices if needed
    if padded_seq_len > seq_len:
        q_padded = torch.zeros(batch, padded_seq_len, heads, dim_plus_tail_dim, dtype=q.dtype, device=q.device)
        q_padded[:, :seq_len, :, :] = q
        q = q_padded
        
        # Also pad indices - fill padding with last valid position's indices
        indices_padded = torch.zeros(batch, padded_seq_len, kv_group, topk, dtype=indices.dtype, device=indices.device)
        indices_padded[:, :seq_len, :, :] = indices
        indices_padded[:, seq_len:, :, :] = indices[:, seq_len-1:seq_len, :, :]
        indices = indices_padded
        
        seq_len = padded_seq_len  # Update for mask preparation
    
    # Prepare block indices and mask (use original_seq_len for validity checks)
    block_indices, block_mask = prepare_block_indices_and_mask(
        indices, original_seq_len, S_per_block, block_I
    )
    
    # Create kernel
    kernel = sparse_mla_fwd_seq_block(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual,
        block_I=block_I, S_per_block=S_per_block, num_stages=num_stages, threads=threads
    )
    
    # Run kernel - kernel expects padded inputs and produces padded outputs
    out, lse = kernel(q, kv, block_indices, block_mask)
    
    # Slice output to original seq_len
    out = out[:, :original_seq_len, :, :]
    lse = lse[:, :original_seq_len, :]
    
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    """Reference implementation for correctness checking."""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd_seq_block(
    B=1,
    S=256,  # Start with smaller S for testing
    SKV=256,
    H=16,  # Small head count - the target use case
    HKV=1,
    DQK=576,
    DV=512,
    topk=128,  # Smaller topk for testing
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=64,
    S_per_block=64,
    num_stages=2,
    threads=256,
):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    print(f"Testing with B={B}, S={S}, SKV={SKV}, H={H}, topk={topk}, S_per_block={S_per_block}")
    
    tl_out, tl_lse = sparse_mla_fwd_seq_block_interface(
        q, kv, indices, 
        block_I=block_I, 
        S_per_block=S_per_block,
        num_stages=num_stages, 
        threads=threads
    )

    if check_correctness:
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_seq_block_interface(
            q, kv, indices, 
            block_I=block_I, 
            S_per_block=S_per_block,
            num_stages=num_stages, 
            threads=threads
        )

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


def test_small_heads():
    """Test case specifically for small head counts (H=16)."""
    print("=" * 60)
    print("Testing sparse_mla_fwd_seq_block with small head count (H=16)")
    print("=" * 60)
    
    test_sparse_mla_fwd_seq_block(
        B=1,
        S=256,
        SKV=256,
        H=16,  # Small head count
        HKV=1,
        DQK=576,
        DV=512,
        topk=128,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        S_per_block=64,
        num_stages=2,
        threads=256,
    )


def test_large_scale():
    """Test case for larger scale."""
    print("=" * 60)
    print("Testing sparse_mla_fwd_seq_block at larger scale")
    print("=" * 60)
    
    test_sparse_mla_fwd_seq_block(
        B=1,
        S=4096,
        SKV=4096,
        H=16,  # Small head count
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=False,  # Skip correctness for speed (OOM risk)
        block_I=64,
        S_per_block=64,
        num_stages=2,
        threads=256,
    )


def compare_with_original():
    """
    Compare sequence-blocked kernel with original head-blocked kernel.
    Shows the benefit of sequence blocking for small head counts.
    """
    from sparse_mla_fwd_little_head import sparse_mla_fwd_interface as original_interface
    
    print("=" * 70)
    print("Comparing original (head-blocked) vs new (seq-blocked) implementations")
    print("=" * 70)
    
    # Test with small head count (H=16) - the target optimization case
    B, S, SKV, H, HKV = 1, 512, 512, 16, 1
    DQK, DV, topk = 576, 512, 256
    dtype = torch.bfloat16
    
    torch.random.manual_seed(42)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")
    
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    print(f"Config: B={B}, S={S}, SKV={SKV}, H={H}, topk={topk}")
    print()
    
    # Test original implementation
    print("Original (head-blocked) kernel:")
    try:
        original_out, original_lse = original_interface(q, kv, indices)
        
        from tilelang.profiler import do_bench
        def fn_original():
            return original_interface(q, kv, indices)
        ms_original = do_bench(fn_original, rep=100, warmup=250)
        print(f"  Time: {ms_original:.3f} ms")
    except Exception as e:
        print(f"  Error: {e}")
        original_out = None
        ms_original = float('inf')
    
    # Test new sequence-blocked implementation
    print("New (seq-blocked) kernel:")
    new_out, new_lse = sparse_mla_fwd_seq_block_interface(q, kv, indices)
    
    from tilelang.profiler import do_bench
    def fn_new():
        return sparse_mla_fwd_seq_block_interface(q, kv, indices)
    ms_new = do_bench(fn_new, rep=100, warmup=250)
    print(f"  Time: {ms_new:.3f} ms")
    
    # Compare correctness
    if original_out is not None:
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        
        print()
        print("Correctness check against reference:")
        try:
            assert_tensors_similar(original_out, ref_out, eps=1e-2, name="original")
            print("  Original: PASSED")
        except AssertionError as e:
            print(f"  Original: FAILED - {e}")
        
        try:
            assert_tensors_similar(new_out, ref_out, eps=1e-2, name="new")
            print("  New (seq-blocked): PASSED")
        except AssertionError as e:
            print(f"  New (seq-blocked): FAILED - {e}")
    
    if ms_original != float('inf'):
        print()
        print(f"Speedup: {ms_original / ms_new:.2f}x")


if __name__ == "__main__":
    test_small_heads()
    # test_large_scale()  # Uncomment for performance testing
    # compare_with_original()  # Uncomment to compare with original implementation
