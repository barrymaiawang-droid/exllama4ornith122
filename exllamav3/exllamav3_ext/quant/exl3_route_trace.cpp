#include "exl3_route_trace.h"
#include "route_trace.h"

#include <ATen/ATen.h>
#include <cstring>

bool exl3_route_trace_enabled()
{
    return exl3_route_trace::Trace::instance().enabled();
}

void exl3_route_trace_set_enabled(bool on)
{
    exl3_route_trace::Trace::instance().set_enabled(on);
}

void exl3_route_trace_reset()
{
    exl3_route_trace::Trace::instance().reset();
}

std::tuple<at::Tensor, int64_t> exl3_route_trace_dump()
{
    using exl3_route_trace::Record;
    using exl3_route_trace::RECORD_INTS;
    int64_t dropped = 0;
    std::vector<Record> records = exl3_route_trace::Trace::instance().snapshot(&dropped);
    at::Tensor out = at::empty({(int64_t) records.size(), (int64_t) RECORD_INTS},
                               at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    if (!records.empty())
        std::memcpy(out.data_ptr<int32_t>(), records.data(), records.size() * sizeof(Record));
    return std::make_tuple(out, dropped);
}
