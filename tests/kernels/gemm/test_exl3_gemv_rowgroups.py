"""RDNA single-matrix verification rows must keep the established AR dot arithmetic.

The 9..16-row boundary previously switched from fdot2 GEMV to cooperative WMMA.
Real Ornith QKV/Z replay reproduced that numerical difference with identical inputs
and A_had. These tests assert row parity, ragged stores, repeat/replay and unchanged
1/8-row controls. They do not replace independent quantization or model quality gates.
All numeric data and comparisons are on the admitted GPU, not a CPU reference experiment.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="RDNA row-group dispatch contract")

CASES = [
    (5, "mul1", 9, 1280, 384, False),
    (5, "mul1", 16, 3072, 12288, True),  # observed Ornith QKV projection
    (5, "mul1", 15, 3072, 8192, True),   # Z shape, ragged final row group
    (2, "mcg", 16, 256, 256, False),
    (3.5, "mul1", 9, 256, 384, True),
    (8, "3inst", 16, 128, 128, True),
    (4, "3inst", 1, 512, 1024, False),  # unchanged AR control
    (4, "mul1", 8, 512, 1024, True),    # unchanged native MTP-sized control
    (3.5, "mul1", 16, 256, 384, False),
    (5, "mul1", 13, 512, 1024, True),  # five valid rows in the final group
]


def operands(device, bits, m, k, n):
    gen = torch.Generator(device=device).manual_seed(1701 + m + k + n)
    # Every packed bit pattern is a valid trellis code; no host quantizer is used.
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, int(16 * bits)),
                       dtype=torch.int16, device=device, generator=gen)
    x = torch.randn((m, k), device=device, dtype=torch.float16, generator=gen) * 0.125
    suh = torch.where(torch.rand(k, device=device, generator=gen) < 0.5, -1.0, 1.0).half()
    svh = torch.where(torch.rand(n, device=device, generator=gen) < 0.5, -1.0, 1.0).half()
    return x, tr, suh, svh


def call(x, tr, suh, svh, y, xh, cb):
    return ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, cb == "mcg", cb == "mul1", 0)


def ar_rows(x, tr, suh, svh, cb, dtype):
    out = torch.empty((x.shape[0], svh.numel()), device=x.device, dtype=dtype)
    had = torch.empty_like(x)
    for i in range(x.shape[0]):
        assert call(x[i:i+1], tr, suh, svh, out[i:i+1], had[i:i+1], cb) == 92
    return out, had


def exact(a, b):
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "AR row arithmetic changed"


@pytest.mark.parametrize("bits,cb,m,k,n,fp32", CASES)
@torch.inference_mode()
def test_rowgroups_match_ar_and_preserve_bounds(device, bits, cb, m, k, n, fp32):
    x, tr, suh, svh = operands(device, bits, m, k, n)
    before = x.clone()
    dtype = torch.float32 if fp32 else torch.float16
    ref, ref_had = ar_rows(x, tr, suh, svh, cb, dtype)
    pad_y, pad_had = 8*n, 8*k
    backing = torch.full((m*n + 2*pad_y,), 7.0, device=device, dtype=dtype)
    had_backing = torch.full((m*k + 2*pad_had,), 7.0, device=device, dtype=torch.float16)
    y = backing[pad_y:-pad_y].view(m, n)
    xh = had_backing[pad_had:-pad_had].view(m, k)
    expected_y = backing.clone()
    expected_had = had_backing.clone()
    expected_y[pad_y:-pad_y].copy_(ref.reshape(-1))
    expected_had[pad_had:-pad_had].copy_(ref_had.reshape(-1))
    for _ in range(2):
        tag = call(x, tr, suh, svh, y, xh, cb)
        assert torch.isfinite(y).all() and torch.isfinite(xh).all()
        exact(y, ref)  # old 9..16-row dispatch is red here, not just on a dispatch label
        exact(xh, ref_had)
        assert tag == 92
        exact(backing, expected_y)
        exact(had_backing, expected_had)
        exact(x, before)


@pytest.mark.parametrize("m,fp32", [(9, False), (16, True)])
@torch.inference_mode()
def test_rowgroups_graph_replay_changes_inputs_and_resets_counters(device, m, fp32):
    x, tr, suh, svh = operands(device, 5, m, 512, 1024)
    dtype = torch.float32 if fp32 else torch.float16
    y = torch.empty((m, 1024), device=device, dtype=dtype)
    xh = torch.empty_like(x)
    original = x.clone()
    # Allocate native per-device parameter storage before graph capture. This is a
    # correctness test, not a first-use timing experiment or model warmup policy.
    call(x, tr, suh, svh, y, xh, "mul1")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tag = call(x, tr, suh, svh, y, xh, "mul1")
    assert tag == 92
    for scale in (1.0, -0.5, 1.0):
        x.copy_(original * scale)
        ref, ref_had = ar_rows(x, tr, suh, svh, "mul1", dtype)
        y.fill_(float("nan"))
        xh.fill_(float("nan"))
        graph.replay()
        exact(y, ref)
        exact(xh, ref_had)


def reference_close(x, tr, suh, svh, y, bits, cb="mul1"):
    # Independent GPU reconstruct + fp32 GEMM, with the existing tail test's
    # fixed 2e-2 relative-RMS operator bound. Not the model's 0.25 quality gate.
    from testlib.exl3 import linear_ref
    ref = linear_ref(x, tr, suh, svh, bits, cb).float()
    assert ref.is_cuda and y.is_cuda
    assert torch.isfinite(y).all() and torch.isfinite(ref).all()
    scale = ref.square().mean().sqrt()
    assert scale > 0
    err = (y.float() - ref).square().mean().sqrt() / scale
    assert err < 2e-2, float(err)


@pytest.mark.parametrize("limit,m,expect_mr", [
    (None, 16, True), ("16", 16, True), ("8", 16, False),
    ("1", 1, True), (None, 17, False),
])
@torch.inference_mode()
def test_rowgroups_route_limits(device, monkeypatch, record_property, limit, m, expect_mr):
    if limit is None:
        monkeypatch.delenv("EXL3_GEMV_MAX_M", raising=False)
    else:
        monkeypatch.setenv("EXL3_GEMV_MAX_M", limit)
    x, tr, suh, svh = operands(device, 5, m, 512, 1024)
    y = torch.empty((m, 1024), device=device, dtype=torch.float32)
    xh = torch.empty_like(x)
    tag = call(x, tr, suh, svh, y, xh, "mul1")
    record_property("dispatch_tag", tag)
    assert (tag == 92) is expect_mr
    reference_close(x, tr, suh, svh, y, 5)
    if expect_mr:
        ref, ref_had = ar_rows(x, tr, suh, svh, "mul1", y.dtype)
        exact(y, ref)
        exact(xh, ref_had)


@torch.inference_mode()
def test_rowgroups_tail_read_at_allocation_end(device):
    m, k, n = 9, 1280, 384
    x, tr, suh, svh = operands(device, 5, m, k, n)
    ref, ref_had = ar_rows(x, tr, suh, svh, "mul1", torch.float32)
    # Same dedicated-allocation tail idiom as test_exl3_gemv_tail.py. No
    # spare rows after the workspace; allocator mapping is not an ASan proof.
    keep = torch.empty(32 << 20, device=device, dtype=torch.float16)
    xh = keep[-m*k:].view(m, k)
    y = torch.empty((m, n), device=device, dtype=torch.float32)
    assert call(x, tr, suh, svh, y, xh, "mul1") == 92
    torch.cuda.synchronize(device)
    exact(y, ref)
    exact(xh, ref_had)


@torch.inference_mode()
def test_multi_matrix_nine_rows_remains_complete(device):
    m, k, n = 9, 256, 384
    x, tr, suh, svh = operands(device, 5, m, k, n)
    # Two distinct matrices detect both slot and ninth-row truncation.
    mats = [(tr, suh, svh), (tr.clone().bitwise_not_(), suh.clone(), -svh)]
    ptrs = lambda tensors: torch.tensor([t.data_ptr() for t in tensors],
                                        device=device, dtype=torch.int64)
    bp, sp, vp = [ptrs([t[i] for t in mats]) for i in range(3)]
    out = torch.full((2, m, n), float("nan"), device=device, dtype=torch.float16)
    had = torch.empty((2, m, k), device=device, dtype=torch.float16)
    ext.exl3_mgemm(x.view(1, m, k), bp, out, sp, had, vp, None, None,
                  5, -1, 0, 1, -1, -1, 0, 1, None, None, None, None, 0)
    for j, (tj, sj, vj) in enumerate(mats):
        reference_close(x, tj, sj, vj, out[j], 5)
