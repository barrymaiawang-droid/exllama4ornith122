#pragma once

// =============================================================================
// Dispatch route trace for the EXL3 GEMM entry points -- diagnostic, default off
// =============================================================================
//
// exl3_gemm_gr and exl3_mgemm_gr choose a kernel per call (RDNA multi-row GEMV, fused m == 1
// GEMV, cooperative GEMM shape, ...) and report the choice only as their int return value,
// which the native callers (BC_* modules under libtorch/) discard. A model-level numerical
// difference therefore cannot be attributed to a route without observing the routes the real
// forward took. This recorder keeps one fixed-width record per call -- shape, bitrate,
// codebook, flags and the returned route -- in host memory, for Python to read back
// (exllamav3.util.route_trace).
//
// Cost model: off, each entry point pays one relaxed atomic load. On, a mutex and a vector
// push per call on the host thread; the device work is untouched. The trace is not timing
// evidence either way: a binary carrying it is a different binary, and an enabled trace adds
// host work per launch. Use it to establish WHICH route a shape took, never how long it took.
//
// Pure C++ (no torch, no CUDA), so it compiles and runs under a plain host compiler; the
// torch-facing accessors are in exl3_route_trace.cpp.
//
// Environment (read once, at first use):
//   EXL3_ROUTE_TRACE      != 0 starts the process with recording on (default off; Python can
//                         toggle it at runtime through exl3_route_trace_set_enabled)
//   EXL3_ROUTE_TRACE_MAX  record capacity (default 262144); further calls are counted as
//                         dropped, not stored, so a long run cannot grow without bound

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace exl3_route_trace
{

enum Kind : int32_t
{
    KIND_GEMM            = 0,   // exl3_gemm_gr (single matrix)
    KIND_MGEMM           = 1,   // exl3_mgemm_gr (multi-matrix / batched pointer tables)
    // The routed-expert tiers do not pass through the two entry points above; without these
    // kinds a route table shows every projection of a layer except its experts
    KIND_MOE_FUSED       = 2,   // exl3_moe: the fused expert kernel (block_sparse_mlp's fused tier,
                                // any batch above MAX_BSZN and prefill); m = tokens, k = hidden,
                                // n = intermediate, slots = active experts in this launch's row
                                // range (-1: unknown, all fused), route = row tile (16 / 32 / 64)
    KIND_MOE_DECODE_RDNA = 3,   // exl3_rocm_moe_decode: the RDNA bsz <= MAX_BSZN decode tier (GEMV
                                // family, M == 1 bodies); m = tokens, k = hidden, n = intermediate,
                                // slots = tokens * top_k, route = 0
    KIND_MOE_COOP        = 4    // exl3_moe_coop_run: the cooperative bsz <= MAX_BSZN decode tier (CUDA,
                                // and the RDNA fallback when moe_decode declines); fields as kind 3
};

enum Flags : int32_t
{
    FLAG_FP32    = 1,    // fp32 output (else fp16)
    FLAG_GRAPH   = 2,    // called with a Graph (capture-time call; replays do not reach the host)
    FLAG_WEIGHTS = 4,    // mgemm: routing weights (weighted reduce)
    FLAG_INDICES = 8,    // mgemm: index table
    FLAG_SLICED  = 16,   // mgemm: sliced mode (had_src_list)
    FLAG_HALF_K  = 32    // half-integer bitrate (tile_u16 % 16 == 8)
};

// Field order is the dump's column order; exllamav3/util/route_trace.py mirrors it (FIELDS)
struct Record
{
    int32_t kind;          // Kind
    int32_t m;             // rows (gemm: product of A's leading dims; mgemm: A.size(1))
    int32_t k;             // input width
    int32_t n;             // output width (mgemm: C.size(2), the slot width)
    int32_t slots;         // mgemm: max(bszm_in, bszm_out); gemm: 1
    int32_t tile_u16;      // trellis tile width in uint16 (16 * bits, + 8 for half rates); MoE
                           // kinds: the gate / up width (the down width is not recorded)
    int32_t cb;            // codebook: 0 3inst, 1 mcg, 2 mul1
    int32_t flags;         // Flags
    int32_t force_shape;   // force_shape_idx argument
    int32_t route;         // the entry point's return value
};

constexpr int RECORD_INTS = 10;
static_assert(sizeof(Record) == RECORD_INTS * sizeof(int32_t), "Record must stay a flat int32 row");

class Trace
{
public:
    static Trace& instance()
    {
        static Trace t;
        return t;
    }

    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
    void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }
    size_t capacity() const { return cap_; }

    void record(const Record& r)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (records_.size() < cap_) records_.push_back(r);
        else ++dropped_;
    }

    // Copy of the records in call order; *dropped receives the count that did not fit
    std::vector<Record> snapshot(int64_t* dropped) const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (dropped) *dropped = dropped_;
        return records_;
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        records_.clear();
        dropped_ = 0;
    }

    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;

private:
    Trace()
    {
        const char* on = std::getenv("EXL3_ROUTE_TRACE");
        enabled_.store(on && std::atoi(on) != 0, std::memory_order_relaxed);
        const char* cap = std::getenv("EXL3_ROUTE_TRACE_MAX");
        long long v = cap ? std::atoll(cap) : 0;
        cap_ = v > 0 ? (size_t) v : (size_t) 262144;
    }

    std::atomic<bool> enabled_;
    size_t cap_;
    mutable std::mutex mtx_;
    std::vector<Record> records_;
    int64_t dropped_ = 0;
};

}  // namespace exl3_route_trace
