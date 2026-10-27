"""
exllamav3/util/exact_linear.py: exact-key linear resolution, diagnostic pointer tables and the
slab layout handed to the entry points, on stub models and a stub extension (no GPU). The
module is loaded from its file so a host without the built extension can run this; the entry
points are stubs that record their arguments, so no route is asserted here.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODULE = os.path.join(REPO, "exllamav3", "util", "exact_linear.py")


@pytest.fixture(scope = "module")
def ex():
    spec = importlib.util.spec_from_file_location("exl3_exact_linear_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod       # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def linear(key, k, n, K = 5, mcg = False, mul1 = True, quant_type = "exl3"):
    inner = SimpleNamespace(
        trellis = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.int16),
        suh = torch.ones(k, dtype = torch.half), svh = torch.ones(n, dtype = torch.half),
        K = K, mcg = mcg, mul1 = mul1, in_features = k, out_features = n,
    )
    return SimpleNamespace(key = key, quant_type = quant_type, inner = inner)


class Model:
    """Model.find_module / iteration as the real Model exposes them"""
    def __init__(self, modules):
        self._modules = {m.key: m for m in modules}

    def find_module(self, key):
        return self._modules[key]

    def __iter__(self):
        return iter(self._modules.values())


L0 = "model.language_model.layers.0.mlp.shared_expert"
L3 = "model.language_model.layers.3.self_attn"


def test_find_linear_exact_key_only(ex):
    gate = linear(f"{L0}.gate_proj", 3072, 1024)
    model = Model([gate, linear(f"{L0}.up_proj", 3072, 1024), SimpleNamespace(key = f"{L0}", quant_type = None)])
    assert ex.find_linear(model, f"{L0}.gate_proj") is gate
    with pytest.raises(ValueError, match = "nearest keys") as e:
        ex.find_linear(model, f"{L0}.gate_pro")      # near miss is still a miss
    assert f"{L0}.gate_proj" in str(e.value)
    with pytest.raises(ValueError, match = "not a loaded EXL3 linear"):
        ex.find_linear(model, L0)                      # the parent module, not a linear
    unloaded = linear(f"{L3}.k_proj", 3072, 512)
    unloaded.inner.trellis = None
    with pytest.raises(ValueError, match = "no trellis"):
        ex.find_linear(Model([unloaded]), f"{L3}.k_proj")


def test_pair_tables_follow_key_order_and_check_compatibility(ex):
    gate, up = linear(f"{L0}.gate_proj", 3072, 1024), linear(f"{L0}.up_proj", 3072, 1024)
    model = Model([up, gate])                          # registration order is not key order
    t = ex.pair_tables(model, [f"{L0}.gate_proj", f"{L0}.up_proj"])
    assert t.keys == (f"{L0}.gate_proj", f"{L0}.up_proj") and t.linears == (gate, up)
    assert t.ptrs_trellis.tolist() == [gate.inner.trellis.data_ptr(), up.inner.trellis.data_ptr()]
    assert t.ptrs_suh.tolist() == [gate.inner.suh.data_ptr(), up.inner.suh.data_ptr()]
    assert t.ptrs_svh.tolist() == [gate.inner.svh.data_ptr(), up.inner.svh.data_ptr()]
    assert t.ptrs_trellis.dtype == torch.long and t.slots == 2 and t.uniform_width
    assert (t.K, t.mcg, t.mul1, t.in_features, t.out_features) == (5, False, True, 3072, (1024, 1024))

    kv = Model([linear(f"{L3}.k_proj", 3072, 512), linear(f"{L3}.v_proj", 3072, 512, K = 4)])
    with pytest.raises(ValueError, match = "does not match"):
        ex.pair_tables(kv, [f"{L3}.k_proj", f"{L3}.v_proj"])
    with pytest.raises(ValueError, match = "no keys"):
        ex.pair_tables(kv, [])


def test_mgemm_pair_allocates_contiguous_slabs_for_this_m(ex, monkeypatch):
    calls = []
    stub = SimpleNamespace(
        exl3_mgemm = lambda *a: calls.append(("mgemm", a)) or 93,
        exl3_gemm = lambda *a: calls.append(("gemm", a)) or 92,
    )
    monkeypatch.setattr(ex, "_ext", lambda: stub)
    gate, up = linear(f"{L0}.gate_proj", 3072, 1024), linear(f"{L0}.up_proj", 3072, 1024)
    t = ex.pair_tables(Model([gate, up]), [f"{L0}.gate_proj", f"{L0}.up_proj"])

    x32 = torch.zeros((32, 3072), dtype = torch.half)
    for m in (1, 8, 16):
        out, had, route = ex.mgemm_pair(x32[:m].contiguous(), t, force_shape_idx = 2, force_num_sms = 20)
        assert route == 93
        assert out.shape == (2, m, 1024) and had.shape == (2, m, 3072)
        assert out.is_contiguous() and had.is_contiguous(), "slot slabs must be contiguous for this m"
        assert out.stride(0) == m * 1024 and had.stride(0) == m * 3072, "a slice of a (2, 32, *) buffer would keep the 32-row slot stride"
        kind, a = calls[-1]
        assert kind == "mgemm" and a[0].shape == (1, m, 3072) and a[2] is out and a[4] is had
        assert a[1] is t.ptrs_trellis and a[3] is t.ptrs_suh and a[5] is t.ptrs_svh
        assert a[6] is None and a[7] is None                       # no indices, no weights
        assert a[8:16] == (5, 2, False, True, -1, -1, 20, 1)        # K, forced shape, mcg, mul1, min, max, sms, num_tokens
        assert a[16:] == (None, None, None, None, 0)                # uniform widths: no per-matrix lists

    with pytest.raises(ValueError, match = "contiguous fp16"):
        ex.mgemm_pair(x32[:8, :1024], t)                            # wrong width / non-contiguous
    with pytest.raises(ValueError, match = "contiguous fp16"):
        ex.mgemm_pair(x32[:8].float(), t)

    out, had, route = ex.gemm_single(x32[3:4].contiguous(), gate, force_shape_idx = 2)
    kind, a = calls[-1]
    assert kind == "gemm" and route == 92 and out.shape == (1, 1024) and had.shape == (1, 3072)
    assert a[1] is gate.inner.trellis and a[3] is gate.inner.suh and a[5] is gate.inner.svh
    assert a[6:] == (2, False, True, 0)


def test_mgemm_pair_mixed_widths_use_per_matrix_destinations(ex, monkeypatch):
    calls = []
    stub = SimpleNamespace(exl3_mgemm = lambda *a: calls.append(a) or 93)
    monkeypatch.setattr(ex, "_ext", lambda: stub)
    a_, b_ = linear("a", 256, 384), linear("b", 256, 256)
    t = ex.pair_tables(Model([a_, b_]), ["a", "b"])
    assert not t.uniform_width
    outs, had, route = ex.mgemm_pair(torch.zeros((9, 256), dtype = torch.half), t, out_dtype = torch.float)
    assert [o.shape for o in outs] == [(9, 384), (9, 256)] and all(o.dtype == torch.float for o in outs)
    a = calls[-1]
    assert a[2].shape == (2, 9, 384)                                # carrier: slots x m x max width
    assert a[16].tolist() == [384, 256] and a[16].dtype == torch.int
    assert a[17].tolist() == [outs[0].data_ptr(), outs[1].data_ptr()]
