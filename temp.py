import torch

from flash_mla import flash_mla_sparse_fwd, flash_mla_sparse_bwd
from sparse_mla_bwd import sparse_mla_bwd_interface


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, q_start_index_s=0):
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
    compressed_casual_mask = torch.arange(q_start_index_s, q_start_index_s + sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
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


def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True, q_start_index_s=0):
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual, q_start_index_s)
    o.backward(do)
    return q.grad, kv.grad

q, kv, out, grad_out, indices, lse, chunk_offset, chunk_offset, offsets, sm_scale = torch.load("/home/users/chenquanlin/workspace/operator/test_smla/smla2.pt")

# q = q[0:1].contiguous()
# out = out[0:1].contiguous()
# grad_out = grad_out[0:1].contiguous()
# indices = indices[0:1].contiguous()
# lse = lse[0:1].contiguous()

# sm_scale=576 ** -0.5

flash_out, _, flash_lse, _ = flash_mla_sparse_fwd(
        q, kv, indices,
        sm_scale=sm_scale, q_start_index_s=0
    )


grad_q1, grad_kv1 = flash_mla_sparse_bwd(
    q, kv, out, grad_out, indices, lse,
    sm_scale=sm_scale,
    q_start_index_s=chunk_offset,
)


log2e = 1.44269504
grad_q, grad_kv = sparse_mla_bwd_interface(
    q,
    kv,
    out,
    grad_out,
    indices,
    lse / log2e,
    offsets,
    chunk_offset=chunk_offset,
    sm_scale=sm_scale,
    return_kernel=False,
    delta=None
)

def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff

print("=== flash vs tilelang ===")
print((grad_q1 - grad_q).abs().max().item())
print((grad_kv1 - grad_kv).abs().max().item())
dq_max_diff, dq_rel_diff = calc_diff(grad_q1, grad_q)
dk_max_diff, dk_rel_diff = calc_diff(grad_kv1, grad_kv)
print(f"dq_max_diff: {dq_max_diff:.6f}, dq_rel_diff: {dq_rel_diff:.6f}")
print(f"dk_max_diff: {dk_max_diff:.6f}, dk_rel_diff: {dk_rel_diff:.6f}")

# ref: autograd-based ground truth (expects 4D inputs with batch dim)
grad_q_ref, grad_kv_ref = ref_sparse_mla_bwd_interface(
    q.unsqueeze(0),
    kv.unsqueeze(0),
    out.unsqueeze(0),
    grad_out.unsqueeze(0).to(torch.bfloat16),
    indices.unsqueeze(0),
    lse=None,
    sm_scale=sm_scale,
    q_start_index_s=chunk_offset,
)
# squeeze batch dim back for comparison
grad_q_ref = grad_q_ref.squeeze(0).to(torch.bfloat16)
grad_kv_ref = grad_kv_ref.squeeze(0).to(torch.bfloat16)

print("\n=== flash vs ref ===")
dq_max_diff, dq_rel_diff = calc_diff(grad_q_ref, grad_q1)
dk_max_diff, dk_rel_diff = calc_diff(grad_kv_ref, grad_kv1)
print(f"dq_max_diff: {dq_max_diff:.6f}, dq_rel_diff: {dq_rel_diff:.6f}")
print(f"dk_max_diff: {dk_max_diff:.6f}, dk_rel_diff: {dk_rel_diff:.6f}")

print("\n=== tilelang vs ref ===")
dq_max_diff, dq_rel_diff = calc_diff(grad_q_ref, grad_q)
dk_max_diff, dk_rel_diff = calc_diff(grad_kv_ref, grad_kv)
print(f"dq_max_diff: {dq_max_diff:.6f}, dq_rel_diff: {dq_rel_diff:.6f}")
print(f"dk_max_diff: {dk_max_diff:.6f}, dk_rel_diff: {dk_rel_diff:.6f}")
