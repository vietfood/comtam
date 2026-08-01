#include <metal_stdlib>
#include "utils.h"

using namespace metal;

template <typename T>
kernel void neg(
    device const T* src0,
    device T* dst,
    constant ViewInfo& view_src0,
    uint id [[thread_position_in_grid]]
) {
    if (id >= view_src0.N) {
        return;
    }
    int64_t src0_pid = physical_offset(id, view_src0);
    dst[id] = -src0[src0_pid];
}

template [[ host_name("neg_fp32") ]]
kernel void neg(
    device const float*,
    device float*,
    constant ViewInfo&,
    uint
);

template <typename T>
kernel void recip(
    device const T* src0,
    device T* dst,
    constant ViewInfo& view_src0,
    uint id [[thread_position_in_grid]]
) {
    if (id >= view_src0.N) {
        return;
    }
    int64_t src0_pid = physical_offset(id, view_src0);
    // Plain division keeps IEEE float32 edges: recip(+/-0) -> +/-inf, recip(+/-inf) -> +/-0.
    dst[id] = T(1) / src0[src0_pid];
}

template [[ host_name("recip_fp32") ]]
kernel void recip(
    device const float*,
    device float*,
    constant ViewInfo&,
    uint
);
