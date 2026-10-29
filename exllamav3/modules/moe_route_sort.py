"""
One-launch expert grouping for small MoE batches (speculative verify windows).

The fused MoE tier needs the (token, expert) assignments grouped by expert: the sort order, the sorted token
ids and weights, per-expert counts, the inverse permutation and the slot tables of the deterministic gather.
BlockSparseMLP builds them from about a dozen small torch ops per layer (argsort, two gathers, bincount,
scatter, cumsum, stack, ...), i.e. hundreds of launches per verify round on a 48-layer model, and the
argsort / bincount kernels are where the decode trace's largest inter-kernel gaps sit on ROCm.

For up to MAX_ASSIGN assignments this module builds all of them in a single Triton program: positions from an
A x A comparison (counting sort, stable), counts from a histogram, starts from a cumsum. The grouping is stable
where argsort is not, which is harmless: every fused-tier output row depends only on its own input row and the
deterministic gather sums each token's slots in top-k order, whatever order the rows of one expert come in.

Opt-in with EXL3_MOE_FAST_SORT=1 while it is being measured on hardware.
"""
from __future__ import annotations
import os
import torch

MAX_ASSIGN = 128
FAST_SORT = os.environ.get("EXL3_MOE_FAST_SORT", "0") != "0"

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_route_sort_kernel(
        sel_ptr, w_ptr,
        order_ptr, token_sorted_ptr, weight_sorted_ptr, inv_order_ptr, count_ptr, tables_ptr,
        A, K, E1,
        A_PAD: tl.constexpr, E_PAD: tl.constexpr,
    ):
        a = tl.arange(0, A_PAD)
        valid = a < A
        e = tl.load(sel_ptr + a, mask = valid, other = 0)
        # Padding lanes sort after every real expert and are never stored
        e_key = tl.where(valid, e, E_PAD)

        # Stable counting-sort position: earlier experts, plus earlier assignments of the same expert
        b = tl.arange(0, A_PAD)
        e_b = tl.where(b < A, tl.load(sel_ptr + b, mask = b < A, other = 0), E_PAD)
        ea = e_key[:, None]
        eb = e_b[None, :]
        before = (eb < ea) | ((eb == ea) & (b[None, :] < a[:, None]))
        pos = tl.sum(before.to(tl.int32), axis = 1)

        w = tl.load(w_ptr + a, mask = valid, other = 0.0)
        tl.store(order_ptr + pos, a.to(tl.int64), mask = valid)
        tl.store(token_sorted_ptr + pos, (a // K).to(tl.int64), mask = valid)
        tl.store(weight_sorted_ptr + pos, w, mask = valid)
        tl.store(inv_order_ptr + a, pos.to(tl.int64), mask = valid)

        # Per-expert counts and starts (start = number of assignments to lower experts)
        ex = tl.arange(0, E_PAD)
        cnt = tl.histogram(tl.where(valid, e, E_PAD - 1).to(tl.int32), E_PAD)
        # The padding lanes were binned into E_PAD - 1, which is never a real expert row (E_PAD > E1)
        pad_lanes = A_PAD - A
        cnt = tl.where(ex == E_PAD - 1, cnt - pad_lanes, cnt)
        start = tl.cumsum(cnt, axis = 0) - cnt
        em = ex < E1
        tl.store(count_ptr + ex, cnt.to(tl.int64), mask = em)
        tl.store(tables_ptr + ex, start.to(tl.int64), mask = em)
        tl.store(tables_ptr + E1 + ex, start.to(tl.int64), mask = em)
        tl.store(tables_ptr + 2 * E1 + ex, (cnt > 0).to(tl.int64), mask = em)

    _kernel = _moe_route_sort_kernel
    return _kernel


def eligible(num_assign: int, num_experts: int) -> bool:
    return FAST_SORT and 0 < num_assign <= MAX_ASSIGN and num_experts + 1 < 1024


def moe_route_sort(selected_experts: torch.Tensor, routing_weights: torch.Tensor, num_experts: int):
    """
    selected_experts (T, K) long in [0, num_experts], routing_weights (T, K) half. Returns
    (order, token_sorted, weight_sorted, expert_count (E+1), inv_order, tables (3, E+1)), matching what
    BlockSparseMLP's all-fused fast path builds from torch ops (up to the order of rows within one expert).
    """
    import triton
    T, K = selected_experts.shape
    A = T * K
    E1 = num_experts + 1
    dev = selected_experts.device
    sel = selected_experts.reshape(-1).contiguous()
    w = routing_weights.reshape(-1).contiguous()
    order = torch.empty((A,), dtype = torch.long, device = dev)
    token_sorted = torch.empty((A,), dtype = torch.long, device = dev)
    weight_sorted = torch.empty((A,), dtype = w.dtype, device = dev)
    inv_order = torch.empty((A,), dtype = torch.long, device = dev)
    expert_count = torch.empty((E1,), dtype = torch.long, device = dev)
    tables = torch.empty((3, E1), dtype = torch.long, device = dev)
    A_PAD = max(16, triton.next_power_of_2(A))
    # One spare bin above every real expert row for the padding lanes
    E_PAD = triton.next_power_of_2(E1 + 1)
    _get_kernel()[(1,)](
        sel, w, order, token_sorted, weight_sorted, inv_order, expert_count, tables,
        A, K, E1, A_PAD = A_PAD, E_PAD = E_PAD, num_warps = 8,
    )
    return order, token_sorted, weight_sorted, expert_count, inv_order, tables
