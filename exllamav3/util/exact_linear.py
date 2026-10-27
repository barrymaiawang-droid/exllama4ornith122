"""
Exact-key linear lookup and diagnostic pointer tables for operator replays.

A same-input / same-weight replay of one projection has to run the exact loaded weights of a
named linear, not a shape-guessed owner, and it has to hand the multi-matrix entry point the
slab layout the kernels assume. Both have tripped up external probes: GatedMLP keeps its
gate/up MultiLinear inside the list `multi_gu` rather than as an attribute, so an attribute
walk never finds it, and a (2, m, *) slice of a (2, 32, *) buffer has the wrong slot stride.

This module resolves linears by their exact module keys through Model.find_module (the model's
own key index; a missing key raises with the nearest keys), builds an explicitly diagnostic
pointer table from those linears (the same layout as MultiLinear's: one trellis / suh / svh
device address per matrix, in the order given), and issues the entry-point calls with freshly
allocated, contiguous (slots, m, *) slabs. Nothing here chooses a route: the entry points'
return values are passed back so the replay can record which family ran.

Typical use (device code; the model is loaded):

    from exllamav3.util import exact_linear as ex
    pair = ex.pair_tables(model, ["model.language_model.layers.0.mlp.shared_expert.gate_proj",
                                  "model.language_model.layers.0.mlp.shared_expert.up_proj"])
    out, had, route = ex.mgemm_pair(x, pair)                   # x: (m, k) fp16, contiguous
    out1, had1, route1 = ex.gemm_single(x[i:i + 1], pair.linears[0])
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Sequence

import torch


def _ext():
    from ..ext import exllamav3_ext
    return exllamav3_ext


def find_linear(model, key: str):
    """The Linear module registered under exactly `key`, which must hold a loaded EXL3 inner"""
    try:
        module = model.find_module(key)
    except KeyError:
        near = difflib.get_close_matches(key, [m.key for m in model], n = 5, cutoff = 0.0)
        raise ValueError(f"no module with key {key!r}; nearest keys: {near}") from None
    inner = getattr(module, "inner", None)
    if getattr(module, "quant_type", None) != "exl3" or inner is None:
        raise ValueError(f"{key!r} is {type(module).__name__} with quant_type {getattr(module, 'quant_type', None)!r}, "
                         f"not a loaded EXL3 linear")
    for attr in ("trellis", "suh", "svh"):
        if getattr(inner, attr, None) is None:
            raise ValueError(f"{key!r}: EXL3 inner has no {attr} tensor (unloaded or swapped out)")
    return module


@dataclass
class PairTables:
    """A diagnostic pointer table over exactly the named linears, in the order given"""
    keys: tuple[str, ...]
    linears: tuple                      # the Linear modules (inner is the LinearEXL3)
    ptrs_trellis: torch.Tensor          # int64 (slots,) device addresses
    ptrs_suh: torch.Tensor
    ptrs_svh: torch.Tensor
    K: float
    mcg: bool
    mul1: bool
    in_features: int
    out_features: tuple[int, ...]
    device: torch.device

    @property
    def slots(self) -> int:
        return len(self.keys)

    @property
    def uniform_width(self) -> bool:
        return len(set(self.out_features)) == 1


def pair_tables(model, keys: Sequence[str]) -> PairTables:
    """
    Pointer tables for the linears at `keys` (two for a gate/up or k/v pair; any count works).
    The matrices must share the input width, bitrate and codebook, as the multi-matrix kernels
    require; the output widths may differ (then mgemm_pair uses per-matrix destinations).
    """
    if not keys:
        raise ValueError("pair_tables: no keys")
    linears = tuple(find_linear(model, k) for k in keys)
    inners = [l.inner for l in linears]
    device = inners[0].trellis.device
    K = inners[0].K
    mcg, mul1 = bool(inners[0].mcg), bool(inners[0].mul1)
    in_features = int(inners[0].in_features)
    for k, i in zip(keys, inners):
        if (i.K, bool(i.mcg), bool(i.mul1), int(i.in_features), i.trellis.device) != (K, mcg, mul1, in_features, device):
            raise ValueError(f"{k!r} (K {i.K}, mcg {bool(i.mcg)}, mul1 {bool(i.mul1)}, in {int(i.in_features)}, "
                             f"{i.trellis.device}) does not match {keys[0]!r} (K {K}, mcg {mcg}, mul1 {mul1}, "
                             f"in {in_features}, {device})")
    def ptrs(attr):
        return torch.tensor([getattr(i, attr).data_ptr() for i in inners], dtype = torch.long, device = device)
    return PairTables(
        keys = tuple(keys), linears = linears,
        ptrs_trellis = ptrs("trellis"), ptrs_suh = ptrs("suh"), ptrs_svh = ptrs("svh"),
        K = K, mcg = mcg, mul1 = mul1, in_features = in_features,
        out_features = tuple(int(i.out_features) for i in inners), device = device,
    )


def _check_input(x: torch.Tensor, in_features: int):
    if x.dim() != 2 or x.dtype != torch.half or not x.is_contiguous() or x.shape[1] != in_features:
        raise ValueError(f"input must be a contiguous fp16 (m, {in_features}) tensor, got {tuple(x.shape)} "
                         f"{x.dtype} contiguous={x.is_contiguous()}")


def mgemm_pair(
    x: torch.Tensor,
    tables: PairTables,
    out_dtype: torch.dtype = torch.half,
    force_shape_idx: int = -1,
    force_num_sms: int = 0,
):
    """
    One multi-matrix call over every matrix of `tables` on the m rows of x: the two-matrix
    shape the native BC_* modules issue (A viewed as (1, m, k), one (slots, m, n) output slab,
    one (slots, m, k) Hadamard slab, both freshly allocated and contiguous for this m).
    Returns (out, had, route). force_shape_idx > 0 pins the cooperative mgemm shape (the WMMA
    family) for a forced-route comparison; -1 lets the dispatcher choose, as the model does.

    Matrices of different widths get per-matrix destinations (size_n_list / c_ptrs); out is
    then a list of (m, n_j) tensors.
    """
    _check_input(x, tables.in_features)
    m = x.shape[0]
    ext = _ext()
    had = torch.empty((tables.slots, m, tables.in_features), dtype = torch.half, device = x.device)
    if tables.uniform_width:
        out = torch.empty((tables.slots, m, tables.out_features[0]), dtype = out_dtype, device = x.device)
        route = ext.exl3_mgemm(
            x.view(1, m, tables.in_features), tables.ptrs_trellis, out, tables.ptrs_suh, had, tables.ptrs_svh,
            None, None, tables.K, force_shape_idx, tables.mcg, tables.mul1, -1, -1, force_num_sms, 1,
            None, None, None, None, 0)
        return out, had, route
    outs = [torch.empty((m, n), dtype = out_dtype, device = x.device) for n in tables.out_features]
    carrier = torch.empty((tables.slots, m, max(tables.out_features)), dtype = out_dtype, device = x.device)
    size_n_list = torch.tensor(tables.out_features, dtype = torch.int, device = x.device)
    c_ptrs = torch.tensor([o.data_ptr() for o in outs], dtype = torch.long, device = x.device)
    route = ext.exl3_mgemm(
        x.view(1, m, tables.in_features), tables.ptrs_trellis, carrier, tables.ptrs_suh, had, tables.ptrs_svh,
        None, None, tables.K, force_shape_idx, tables.mcg, tables.mul1, -1, -1, force_num_sms, 1,
        size_n_list, c_ptrs, None, None, 0)
    return outs, had, route


def gemm_single(
    x: torch.Tensor,
    linear,
    out_dtype: torch.dtype = torch.half,
    force_shape_idx: int = -1,
    force_num_sms: int = 0,
):
    """
    The single-matrix call on the m rows of x against one linear's exact weights, with fresh
    contiguous (m, n) output and (m, k) Hadamard slabs. Returns (out, had, route).
    """
    inner = linear.inner
    _check_input(x, int(inner.in_features))
    m = x.shape[0]
    out = torch.empty((m, int(inner.out_features)), dtype = out_dtype, device = x.device)
    had = torch.empty_like(x)
    route = _ext().exl3_gemm(x, inner.trellis, out, inner.suh, had, inner.svh, force_shape_idx,
                             bool(inner.mcg), bool(inner.mul1), force_num_sms)
    return out, had, route
