"""
Dispatch route trace for the EXL3 GEMM entry points (diagnostic, default off).

exl3_gemm / exl3_gemm_gr and exl3_mgemm / exl3_mgemm_gr pick a kernel per call and return the
choice as an int that the native callers (the BC_* modules) discard, so nothing in a model
forward records which route a given projection took. When a model-level numerical difference
is suspected to follow the route (RDNA multi-row GEMV vs cooperative WMMA GEMM, split-K vs
single-warp GEMV, ...), the attribution needs the routes the real forward took, at the real
shapes, under the real configuration -- not an inference from batch size or prompt length.

Usage, around the pass under investigation:

    from exllamav3.util import route_trace
    with route_trace.capture() as t:
        logits = model.forward(ids, params)
    print(t.table())           # one line per distinct (entry, shape, bitrate, flags, route)
    t.records                  # the raw calls, in order

or set EXL3_ROUTE_TRACE=1 in the environment to record from process start and call
snapshot() / summarize() / format_table() later. EXL3_ROUTE_TRACE_MAX (default 262144)
bounds the record count; calls past it are counted in `dropped`, not stored.

What a record is: one host-side entry-point call. A graph-captured call is recorded once, at
capture time (flag GRAPH), and its replays are not calls. The kernels and their arithmetic are
untouched by the trace; the only cost is host work per call while recording. The trace is not
timing evidence in either state: a binary carrying it is a different binary from one without.
"""
from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Iterable, NamedTuple, Sequence

KIND_GEMM = 0
KIND_MGEMM = 1
KIND_NAMES = {KIND_GEMM: "gemm", KIND_MGEMM: "mgemm"}

FLAG_FP32 = 1
FLAG_GRAPH = 2
FLAG_WEIGHTS = 4
FLAG_INDICES = 8
FLAG_SLICED = 16
FLAG_HALF_K = 32
FLAG_NAMES = (
    (FLAG_FP32, "fp32"),
    (FLAG_GRAPH, "graph"),
    (FLAG_WEIGHTS, "weights"),
    (FLAG_INDICES, "indices"),
    (FLAG_SLICED, "sliced"),
    (FLAG_HALF_K, "half_k"),
)

CB_NAMES = {0: "3inst", 1: "mcg", 2: "mul1"}

# Column order of the native dump; must match struct Record in exllamav3_ext/quant/route_trace.h
FIELDS = ("kind", "m", "k", "n", "slots", "tile_u16", "cb", "flags", "force_shape", "route")


class Record(NamedTuple):
    kind: int
    m: int
    k: int
    n: int
    slots: int
    tile_u16: int
    cb: int
    flags: int
    force_shape: int
    route: int

    @property
    def bits(self) -> float:
        return self.tile_u16 / 16

    @property
    def flag_names(self) -> tuple[str, ...]:
        return tuple(name for bit, name in FLAG_NAMES if self.flags & bit)

    @property
    def key(self) -> tuple:
        """Everything but position: records with equal keys are the same dispatch decision"""
        return tuple(self)

    @property
    def route_name(self) -> str:
        return route_name(self.kind, self.route)


class Summary(NamedTuple):
    record: Record     # representative (all grouped records are equal field-for-field)
    count: int
    first: int         # index of the first call in the trace
    last: int          # index of the last call in the trace


def route_name(kind: int, route: int) -> str:
    """
    Names for the entry points' return values, taken from exl3_gemm.cu. The values are the
    kernel family only: a shape index says which cooperative GEMM shape was launched, nothing
    about its tuning.
    """
    if kind == KIND_GEMM:
        if route == 92: return "rdna_multirow_gemv"          # rocm/quant/exl3_gemv_multirow_rdna.cu
        if route == 91: return "rdna_gemv_graph"             # rocm/quant/exl3_gemv_rdna.cu, graph form
        if route == 90: return "gemv"                        # exl3_gemv_try_launch (CUDA GEMV / RDNA m == 1)
        if route == 0:  return "none_or_int8_gemv"           # empty shape, int8 GEMV, or no kernel
        if route >= 1:  return f"coop_gemm_shape_{route}"
    elif kind == KIND_MGEMM:
        if route == 93: return "rdna_multirow_mgemv"         # exl3_mgemv_multirow_try_launch
        if route == 0:  return "none_or_rdna_mgemv_m1"       # empty selection, or exl3_mgemv_try_launch
        if route >= 1:  return f"coop_mgemm_shape_{route}"
    return f"route_{route}"


def _ext():
    from ..ext import exllamav3_ext
    return exllamav3_ext


def available() -> bool:
    """True when the loaded extension carries the trace (a binary built from this source)"""
    return hasattr(_ext(), "exl3_route_trace_dump")


def enabled() -> bool:
    return bool(_ext().exl3_route_trace_enabled())


def set_enabled(on: bool):
    _ext().exl3_route_trace_set_enabled(bool(on))


def reset():
    _ext().exl3_route_trace_reset()


def snapshot() -> tuple[list[Record], int]:
    """(records in call order, dropped count) since the last reset"""
    rows, dropped = _ext().exl3_route_trace_dump()
    return records_from_rows(rows.tolist()), int(dropped)


def records_from_rows(rows: Iterable[Sequence[int]]) -> list[Record]:
    out = []
    for row in rows:
        if len(row) != len(FIELDS):
            raise ValueError(f"route trace row has {len(row)} fields, expected {len(FIELDS)}")
        out.append(Record(*(int(v) for v in row)))
    return out


def summarize(records: Sequence[Record]) -> list[Summary]:
    """Group equal dispatch decisions, in order of first appearance"""
    groups: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for i, r in enumerate(records):
        groups.setdefault(r.key, []).append(i)
    return [Summary(records[idx[0]], len(idx), idx[0], idx[-1]) for idx in groups.values()]


def format_table(summaries: Sequence[Summary], dropped: int = 0) -> str:
    header = ("entry", "m", "k", "n", "slots", "bits", "cb", "flags", "force", "route", "name", "count", "first", "last")
    rows = [header]
    for s in summaries:
        r = s.record
        rows.append((
            KIND_NAMES.get(r.kind, str(r.kind)), str(r.m), str(r.k), str(r.n), str(r.slots),
            f"{r.bits:g}", CB_NAMES.get(r.cb, str(r.cb)), ",".join(r.flag_names) or "-",
            str(r.force_shape), str(r.route), r.route_name, str(s.count), str(s.first), str(s.last),
        ))
    widths = [max(len(row[c]) for row in rows) for c in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    if dropped:
        lines.append(f"({dropped} calls past the capacity were dropped; raise EXL3_ROUTE_TRACE_MAX)")
    return "\n".join(lines)


class Captured:
    """Result of a capture(): the records of the enclosed calls, in order"""

    def __init__(self):
        self.records: list[Record] = []
        self.dropped: int = 0

    def summary(self) -> list[Summary]:
        return summarize(self.records)

    def table(self) -> str:
        return format_table(self.summary(), self.dropped)

    def select(self, **fields) -> list[Record]:
        """Records whose named fields equal the given values, e.g. select(kind=KIND_GEMM, m=16)"""
        return [r for r in self.records if all(getattr(r, f) == v for f, v in fields.items())]


@contextlib.contextmanager
def capture():
    """
    Record every entry-point call inside the block. Clears earlier records, enables the trace
    for the block and restores the previous enabled state afterwards; the Captured object is
    filled on exit.
    """
    ext = _ext()
    if not hasattr(ext, "exl3_route_trace_dump"):
        raise RuntimeError("the loaded exllamav3_ext was built without the route trace")
    was = bool(ext.exl3_route_trace_enabled())
    cap = Captured()
    ext.exl3_route_trace_reset()
    ext.exl3_route_trace_set_enabled(True)
    try:
        yield cap
    finally:
        ext.exl3_route_trace_set_enabled(was)
        rows, dropped = ext.exl3_route_trace_dump()
        cap.records = records_from_rows(rows.tolist())
        cap.dropped = int(dropped)
