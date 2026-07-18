#include <metal_stdlib>
#include "utils.h"

using namespace metal;

/* --- Reduction sum Kernel --- */
template<typename T>
kernel void reduce_sum_full(
    device const T* src0,
    device T* dst,
    constant ViewInfo& view_src0,
    uint2 block_pos [[ threadgroup_position_in_grid ]],
    uint2 thread_pos [[ thread_position_in_threadgroup ]]
) {
}
