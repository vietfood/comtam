#include <metal_stdlib>
#include "utils.h"

using namespace metal;

/* --- Add Kernel --- */
// https://stackoverflow.com/questions/56687496/how-to-make-templated-compute-kernels-in-metal
template<typename T>
kernel void add(
    device const T* src0,
    device const T* src1,
    device T* dst,
    constant ViewInfo& view_src0,
    constant ViewInfo& view_src1,
    uint id [[thread_position_in_grid]]
) {
    // we assume M = 1 for binary kernels
    if (id >= view_src0.N) {
        return;
    }
    // convert from linear index => physical offset
    int64_t src0_pid = physical_offset(id, view_src0);
    int64_t src1_pid = physical_offset(id, view_src1);
    // do calculation
    dst[id] = src0[src0_pid] + src1[src1_pid];
}

template [[ host_name("add_fp32") ]]
kernel void add(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    uint
);

/* --- Sub Kernel --- */
template <typename T>
kernel void sub(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant ViewInfo& view_src0,
        constant ViewInfo& view_src1,
        uint id [[thread_position_in_grid]]) {
    if (id >= view_src0.N) {
        return;
    }
    int64_t src0_pid = physical_offset(id, view_src0);
    int64_t src1_pid = physical_offset(id, view_src1);
    dst[id] = src0[src0_pid] - src1[src1_pid];
}

template [[ host_name("sub_fp32") ]]
kernel void sub(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    uint
);

/* -- Mul Kernel --- */
template <typename T>
kernel void mul(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant ViewInfo& view_src0,
        constant ViewInfo& view_src1,
        uint id [[thread_position_in_grid]]) {
    if (id >= view_src0.N) {
        return;
    }
    int64_t src0_pid = physical_offset(id, view_src0);
    int64_t src1_pid = physical_offset(id, view_src1);
    dst[id] = src0[src0_pid] * src1[src1_pid];
}

template [[ host_name("mul_fp32") ]]
kernel void mul(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    uint
);

/* -- Div Kernel --- */
template <typename T>
kernel void div(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant ViewInfo& view_src0,
        constant ViewInfo& view_src1,
        uint id [[thread_position_in_grid]]) {
    if (id >= view_src0.N) {
        return;
    }
    int64_t src0_pid = physical_offset(id, view_src0);
    int64_t src1_pid = physical_offset(id, view_src1);
    if (src1[src1_pid] == 0) {
        dst[id] = INFINITY;
    } else {
        dst[id] = src0[src0_pid] / src1[src1_pid];
    }
}

template [[ host_name("div_fp32") ]]
kernel void div(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    uint
);
