#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <tuple>

// Python-facing accessors of the dispatch route trace (quant/route_trace.h). See
// exllamav3/util/route_trace.py for the consumer.

bool exl3_route_trace_enabled();
void exl3_route_trace_set_enabled(bool on);
void exl3_route_trace_reset();
// (records, dropped): records is an int32 CPU tensor of shape (count, RECORD_INTS) in call
// order, one row per entry-point call recorded since the last reset; dropped counts the
// calls that did not fit the capacity (EXL3_ROUTE_TRACE_MAX)
std::tuple<at::Tensor, int64_t> exl3_route_trace_dump();
