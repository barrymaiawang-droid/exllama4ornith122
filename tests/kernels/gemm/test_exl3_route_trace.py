"""
Dispatch route trace (exllamav3_ext/quant/route_trace.h, exllamav3/util/route_trace.py): the
host-side recorder and its Python consumer, without a GPU and without the built extension.

The recorder is pure C++ and is compiled here with the host compiler; the Python side is fed
the rows the native dump would produce. Neither checks which route a real call takes -- that
is the trace's job on the admitted GPU, and it is deliberately not asserted from a CPU host.
The consumer module is loaded from its file so a host without the built extension (which the
package import needs) can still run this.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HEADER = os.path.join(REPO, "exllamav3", "exllamav3_ext", "quant", "route_trace.h")
MODULE = os.path.join(REPO, "exllamav3", "util", "route_trace.py")


@pytest.fixture(scope = "module")
def rt():
    spec = importlib.util.spec_from_file_location("exl3_route_trace_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRows:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(r) for r in self.rows]


def fake_ext(rows):
    """The four native accessors over a list of rows, with the enable/reset state they keep"""
    st = SimpleNamespace(enabled = False, resets = 0)
    st.exl3_route_trace_enabled = lambda: st.enabled
    st.exl3_route_trace_set_enabled = lambda on: setattr(st, "enabled", bool(on))

    def reset():
        st.resets += 1
        rows.clear()
    st.exl3_route_trace_reset = reset
    st.exl3_route_trace_dump = lambda: (FakeRows(rows), 0)
    return st


# kind, m, k, n, slots, tile_u16, cb, flags, force_shape, route
QKV_M16_MR   = (0, 16, 3072, 12288, 1, 80, 2, 1, -1, 92)
QKV_M16_WMMA = (0, 16, 3072, 12288, 1, 80, 2, 1, -1, 2)
Z_M1_GEMV    = (0, 1, 3072, 8192, 1, 80, 2, 1, -1, 90)
MOE_WEIGHTED = (1, 1, 3072, 1536, 16, 80, 2, 1 | 4 | 8 | 2, -1, 93)
HEAD_HALF    = (0, 1, 3072, 248320, 1, 56, 2, 32, -1, 90)


def test_field_order_matches_native_record(rt):
    src = open(HEADER).read()
    ints = int(re.search(r"constexpr int RECORD_INTS = (\d+);", src).group(1))
    assert ints == len(rt.FIELDS) == len(rt.Record._fields)
    body = src[src.index("struct Record"):src.index("constexpr int RECORD_INTS")]
    native_fields = tuple(re.findall(r"int32_t (\w+);", body))
    assert native_fields == rt.FIELDS, "struct Record and FIELDS must list the same columns in the same order"
    for bit, name in rt.FLAG_NAMES:
        assert re.search(rf"FLAG_{name.upper()}\s*=\s*{bit}\b", src), f"FLAG_{name.upper()} = {bit} missing from the header"


def test_records_summary_and_table(rt):
    rows = [QKV_M16_WMMA, Z_M1_GEMV, QKV_M16_WMMA, MOE_WEIGHTED, QKV_M16_MR, HEAD_HALF]
    records = rt.records_from_rows(rows)
    assert records[0].bits == 5 and records[0].route_name == "coop_gemm_shape_2"
    assert records[4].route_name == "rdna_multirow_gemv"
    assert records[3].flag_names == ("fp32", "graph", "weights", "indices")
    assert records[5].bits == 3.5 and records[5].flag_names == ("half_k",)

    summary = rt.summarize(records)
    assert [(s.record.route, s.count, s.first, s.last) for s in summary] == [
        (2, 2, 0, 2), (90, 1, 1, 1), (93, 1, 3, 3), (92, 1, 4, 4), (90, 1, 5, 5),
    ]
    # The two 16-row QKV calls that took different routes are different lines: the route is
    # part of the key, so a summary never hides a route change behind a shape
    assert len({s.record.key for s in summary}) == 5

    table = rt.format_table(summary, dropped = 3)
    lines = table.splitlines()
    assert lines[0].split()[:4] == ["entry", "m", "k", "n"]
    assert any("12288" in l and "coop_gemm_shape_2" in l and l.split()[-3:] == ["2", "0", "2"] for l in lines)
    assert any("mgemv" in l and "fp32,graph,weights,indices" in l for l in lines)
    assert lines[-1].startswith("(3 calls past the capacity")

    with pytest.raises(ValueError):
        rt.records_from_rows([(0, 1, 2)])


def test_capture_enables_resets_and_restores(rt, monkeypatch):
    rows = [QKV_M16_WMMA]           # stale record from before the block: must be cleared
    ext = fake_ext(rows)
    monkeypatch.setattr(rt, "_ext", lambda: ext)
    assert rt.available() and not rt.enabled()

    with rt.capture() as cap:
        assert ext.enabled and ext.resets == 1 and rows == []
        rows.extend([Z_M1_GEMV, QKV_M16_MR])      # what the entry points would have recorded
    assert not ext.enabled, "capture must restore the previous enabled state"
    assert [r.route for r in cap.records] == [90, 92]
    assert cap.select(kind = rt.KIND_GEMM, m = 16) == [rt.Record(*QKV_M16_MR)]
    assert "rdna_multirow_gemv" in cap.table()

    # A previously enabled trace stays enabled afterwards
    rt.set_enabled(True)
    with rt.capture():
        pass
    assert rt.enabled()
    rt.reset()
    assert ext.resets == 3

    monkeypatch.setattr(rt, "_ext", lambda: SimpleNamespace())
    assert not rt.available()
    with pytest.raises(RuntimeError):
        with rt.capture():
            pass


HARNESS = r"""
#include "route_trace.h"
#include <cstdio>
using namespace exl3_route_trace;
int main()
{
    Trace& t = Trace::instance();
    printf("enabled %d cap %zu\n", (int) t.enabled(), t.capacity());
    Record r{KIND_GEMM, 0, 3072, 12288, 1, 80, 2, FLAG_FP32, -1, 92};
    t.set_enabled(true);
    for (int i = 0; i < 6; ++i) { r.m = 9 + i; t.record(r); }
    int64_t dropped = -1;
    std::vector<Record> s = t.snapshot(&dropped);
    printf("stored %zu dropped %lld\n", s.size(), (long long) dropped);
    for (const Record& x : s)
        printf("row %d %d %d %d %d %d %d %d %d %d\n", x.kind, x.m, x.k, x.n, x.slots, x.tile_u16, x.cb, x.flags, x.force_shape, x.route);
    t.reset();
    s = t.snapshot(&dropped);
    printf("after_reset %zu dropped %lld\n", s.size(), (long long) dropped);
    printf("sizeof %zu ints %d\n", sizeof(Record), RECORD_INTS);
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason = "needs a host C++ compiler")
@pytest.mark.parametrize("env, expect_enabled, expect_cap, expect_stored, expect_dropped", [
    ({}, 0, 262144, 6, 0),                                              # default: off, large capacity
    ({"EXL3_ROUTE_TRACE": "1", "EXL3_ROUTE_TRACE_MAX": "4"}, 1, 4, 4, 2),  # env start + capacity bound
    ({"EXL3_ROUTE_TRACE": "0"}, 0, 262144, 6, 0),                       # explicit off
])
def test_recorder_host_build(rt, tmp_path, env, expect_enabled, expect_cap, expect_stored, expect_dropped):
    src = tmp_path / "harness.cpp"
    src.write_text(HARNESS)
    exe = tmp_path / "harness"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", os.path.dirname(HEADER), str(src), "-o", str(exe)],
        check = True)
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_ROUTE_TRACE")}
    run_env.update(env)
    out = subprocess.run([str(exe)], check = True, capture_output = True, text = True, env = run_env).stdout
    assert f"enabled {expect_enabled} cap {expect_cap}" in out
    assert f"stored {expect_stored} dropped {expect_dropped}" in out
    rows = [list(map(int, l.split()[1:])) for l in out.splitlines() if l.startswith("row ")]
    records = rt.records_from_rows(rows)      # the dump's rows parse through the Python side unchanged
    assert [r.m for r in records] == list(range(9, 9 + expect_stored)), "records keep call order; the overflow is dropped, not wrapped"
    assert all(r.route == 92 and r.n == 12288 for r in records)
    assert "after_reset 0 dropped 0" in out
    assert f"sizeof {4 * len(rt.FIELDS)} ints {len(rt.FIELDS)}" in out
