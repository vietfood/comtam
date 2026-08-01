#include <metal_stdlib>
#include "utils.h"

using namespace metal;

/*
 * CUDA-style tree reductions, one threadgroup per output:
 *   - each thread strides over the reduction extent, accumulating a partial
 *   - tree-reduce the partials in threadgroup memory
 *   - thread 0 writes the result
 * The host must dispatch a power-of-two threadgroup size equal to BLOCK_SIZE
 * so the tree sweep is well defined.
 */

#define REDUCE_BLOCK_SIZE 256u

template <typename T>
struct SumOp {
    static T init() { return T(0); }
    static T combine(T a, T b) { return a + b; }
};

template <typename T>
struct MaxOp {
    static T init() { return -INFINITY; }
    static T combine(T a, T b) { return a > b ? a : b; }
};

/* --- Full reduce: one threadgroup reduces all N elements to dst[0] --- */
template <typename T, typename Op>
kernel void reduce_full(
    device const T* src,
    device T* dst,
    constant ViewInfo& view_src0,
    uint tid [[thread_position_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]
) {
    threadgroup T smem[REDUCE_BLOCK_SIZE];

    T acc = Op::init();
    for (uint i = tid; i < view_src0.N; i += thread_count) {
        int64_t pid = physical_offset(i, view_src0);
        acc = Op::combine(acc, src[pid]);
    }
    smem[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = REDUCE_BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = Op::combine(smem[tid], smem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        dst[0] = smem[0];
    }
}

/* --- Axis reduce: one threadgroup per output element --- */
template <typename T, typename Op>
kernel void reduce_axis(
    device const T* src,
    device T* dst,
    constant ViewInfo& view_src0,
    constant int& axis,
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 thread_count [[threads_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
    threadgroup T smem[REDUCE_BLOCK_SIZE];

    const uint t = tid.x;
    const uint tc = thread_count.x;
    const uint out_index = gid.x;
    const int64_t red_size = view_src0.shape[axis];

    T acc = Op::init();
    for (uint ri = t; ri < red_size; ri += tc) {
        int64_t pid = physical_offset_axis(out_index, ri, axis, view_src0);
        acc = Op::combine(acc, src[pid]);
    }
    smem[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = REDUCE_BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            smem[t] = Op::combine(smem[t], smem[t + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        dst[out_index] = smem[0];
    }
}

/* --- fp32 instantiations --- */
template [[host_name("reduce_sum_full_fp32")]]
kernel void reduce_full<float, SumOp<float>>(
    device const float*, device float*, constant ViewInfo&, uint, uint);

template [[host_name("reduce_max_full_fp32")]]
kernel void reduce_full<float, MaxOp<float>>(
    device const float*, device float*, constant ViewInfo&, uint, uint);

template [[host_name("reduce_sum_axis_fp32")]]
kernel void reduce_axis<float, SumOp<float>>(
    device const float*, device float*, constant ViewInfo&, constant int&, uint3, uint3, uint3);

template [[host_name("reduce_max_axis_fp32")]]
kernel void reduce_axis<float, MaxOp<float>>(
    device const float*, device float*, constant ViewInfo&, constant int&, uint3, uint3, uint3);
