"""RDNA multi-matrix 9..16-row calls (EXL3_MGEMV_MAX_M=16) must keep the m == 1 multi-row arithmetic per row.

The two-matrix shapes of a verify step (gate/up and k/v pairs, K3072 / N1024 or N512, 5-bit mul1, fp16) take
the RDNA multi-row mgemv (route 93) at m <= 8 and fell to the cooperative WMMA mgemm (a shape index) above,
the last traced AR-vs-verification family difference among the dense projections. With EXL3_MGEMV_MAX_M=16
each slot runs as two <= 8-row groups with its own counters; these tests assert, per slot, byte parity with
separate m == 1 route-93 calls, ragged final groups, sentinel-padded slabs, per-matrix destinations
(size_n_list / c_ptrs), graph replay with changing inputs, and that the default still declines above 8 rows.
Operator contract only: not model-quality evidence. Numeric data and comparisons are on the admitted GPU.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="RDNA multi-row mgemv dispatch contract")

# bits, cb, m, k, widths, fp32
CASES = [
    (5, "mul1", 16, 3072, (1024, 1024), False),   # observed gate/up pair shape
    (5, "mul1", 16, 3072, (512, 512), False),     # observed k/v pair shape
    (5, "mul1", 9, 256, (384, 384), False),       # one valid row in the final group
    (5, "mul1", 13, 512, (512, 512), True),       # five valid rows in the final group
    (2, "mcg", 16, 256, (256, 256), False),
    (3.5, "mul1", 12, 256, (384, 384), True),     # half-integer rate
    (4, "3inst", 8, 512, (1024, 1024), True),     # unchanged <= 8-row control (one group)
    (5, "mul1", 16, 256, (384, 256), False),      # per-matrix destinations
]


def operands(device, bits, m, k, widths):
    gen = torch.Generator(device=device).manual_seed(2201 + m + k + sum(widths))
    mats = []
    for n in widths:
        tr = torch.randint(-32768, 32768, (k // 16, n // 16, int(16 * bits)), dtype=torch.int16, device=device, generator=gen)
        suh = torch.where(torch.rand(k, device=device, generator=gen) < 0.5, -1.0, 1.0).half()
        svh = torch.where(torch.rand(n, device=device, generator=gen) < 0.5, -1.0, 1.0).half()
        mats.append((tr, suh, svh))
    x = torch.randn((m, k), device=device, dtype=torch.float16, generator=gen) * 0.125
    return x, mats


def ptrs(tensors, device):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device=device)


def call(x, mats, out, had, bits, cb, size_n_list=None, c_ptrs=None):
    m, k = x.shape
    bp, sp, vp = [ptrs([t[i] for t in mats], x.device) for i in range(3)]
    return ext.exl3_mgemm(x.view(1, m, k), bp, out, sp, had, vp, None, None, bits, -1,
                          cb == "mcg", cb == "mul1", -1, -1, 0, 1, size_n_list, c_ptrs, None, None, 0)


def exact(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "per-row multi-row arithmetic changed"


def ar_rows(x, mats, bits, cb, dtype, widths):
    """Separate m == 1 route-93 calls, one per row, as (slots, m, n) and (slots, m, k) slabs"""
    m, k = x.shape
    uniform = len(set(widths)) == 1
    out = [torch.empty((m, n), device=x.device, dtype=dtype) for n in widths]
    had = torch.empty((len(mats), m, k), device=x.device, dtype=torch.half)
    for i in range(m):
        h = torch.empty((len(mats), 1, k), device=x.device, dtype=torch.half)
        if uniform:
            o = torch.empty((len(mats), 1, widths[0]), device=x.device, dtype=dtype)
            assert call(x[i:i+1], mats, o, h, bits, cb) == 93
            for j in range(len(mats)):
                out[j][i].copy_(o[j, 0])
        else:
            os_ = [torch.empty((1, n), device=x.device, dtype=dtype) for n in widths]
            carrier = torch.empty((len(mats), 1, max(widths)), device=x.device, dtype=dtype)
            nl = torch.tensor(widths, dtype=torch.int, device=x.device)
            cp = ptrs(os_, x.device)
            assert call(x[i:i+1], mats, carrier, h, bits, cb, nl, cp) == 93
            for j in range(len(mats)):
                out[j][i].copy_(os_[j][0])
        had[:, i].copy_(h[:, 0])
    return out, had


@pytest.mark.parametrize("bits,cb,m,k,widths,fp32", CASES)
@torch.inference_mode()
def test_mgemm_rowgroups_match_m1_rows(device, monkeypatch, bits, cb, m, k, widths, fp32):
    monkeypatch.setenv("EXL3_MGEMV_MAX_M", "16")
    x, mats = operands(device, bits, m, k, widths)
    before = x.clone()
    dtype = torch.float32 if fp32 else torch.float16
    ref, ref_had = ar_rows(x, mats, bits, cb, dtype, widths)
    slots = len(mats)
    uniform = len(set(widths)) == 1
    pad = 8 * max(widths)
    had_backing = torch.full((slots * m * k + 2 * pad,), 7.0, device=device, dtype=torch.float16)
    had = had_backing[pad:-pad].view(slots, m, k)
    expected_had = had_backing.clone()
    expected_had[pad:-pad].copy_(ref_had.reshape(-1))
    for _ in range(2):
        if uniform:
            backing = torch.full((slots * m * widths[0] + 2 * pad,), 7.0, device=device, dtype=dtype)
            out = backing[pad:-pad].view(slots, m, widths[0])
            tag = call(x, mats, out, had, bits, cb)
            assert tag == 93
            for j in range(slots):
                assert torch.isfinite(out[j]).all()
                exact(out[j], ref[j])
            expected = backing.clone()
            expected[pad:-pad].copy_(torch.stack(ref).reshape(-1))
            exact(backing, expected)
        else:
            backings = [torch.full((m * n + 2 * pad,), 7.0, device=device, dtype=dtype) for n in widths]
            outs = [b[pad:-pad].view(m, n) for b, n in zip(backings, widths)]
            carrier = torch.empty((slots, m, max(widths)), device=device, dtype=dtype)
            nl = torch.tensor(widths, dtype=torch.int, device=device)
            tag = call(x, mats, carrier, had, bits, cb, nl, ptrs(outs, device))
            assert tag == 93
            for j in range(slots):
                assert torch.isfinite(outs[j]).all()
                exact(outs[j], ref[j])
                expected = backings[j].clone()
                expected[pad:-pad].copy_(ref[j].reshape(-1))
                exact(backings[j], expected)
        exact(had, ref_had)
        exact(had_backing, expected_had)
        exact(x, before)


@pytest.mark.parametrize("limit,m,expect_mr", [
    (None, 16, False), ("8", 16, False), ("16", 16, True), ("16", 9, True), ("16", 8, True), ("12", 13, False),
])
@torch.inference_mode()
def test_mgemm_rowgroups_opt_in(device, monkeypatch, limit, m, expect_mr):
    if limit is None:
        monkeypatch.delenv("EXL3_MGEMV_MAX_M", raising=False)
    else:
        monkeypatch.setenv("EXL3_MGEMV_MAX_M", limit)
    x, mats = operands(device, 5, m, 512, (1024, 1024))
    out = torch.empty((2, m, 1024), device=device, dtype=torch.float16)
    had = torch.empty((2, m, 512), device=device, dtype=torch.float16)
    tag = call(x, mats, out, had, 5, "mul1")
    assert (tag == 93) is expect_mr
    assert torch.isfinite(out).all()
    if expect_mr:
        ref, ref_had = ar_rows(x, mats, 5, "mul1", torch.float16, (1024, 1024))
        for j in range(2):
            exact(out[j], ref[j])
        exact(had, ref_had)


@pytest.mark.parametrize("m,fp32", [(9, False), (16, True)])
@torch.inference_mode()
def test_mgemm_rowgroups_graph_replay(device, monkeypatch, m, fp32):
    monkeypatch.setenv("EXL3_MGEMV_MAX_M", "16")
    x, mats = operands(device, 5, m, 512, (1024, 1024))
    dtype = torch.float32 if fp32 else torch.float16
    out = torch.empty((2, m, 1024), device=device, dtype=dtype)
    had = torch.empty((2, m, 512), device=device, dtype=torch.float16)
    original = x.clone()
    call(x, mats, out, had, 5, "mul1")       # parameter block allocation before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tag = call(x, mats, out, had, 5, "mul1")
    assert tag == 93
    for scale in (1.0, -0.5, 1.0):
        x.copy_(original * scale)
        ref, ref_had = ar_rows(x, mats, 5, "mul1", dtype, (1024, 1024))
        out.fill_(float("nan"))
        had.fill_(float("nan"))
        graph.replay()
        for j in range(2):
            exact(out[j], ref[j])
        exact(had, ref_had)
