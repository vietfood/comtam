#include <metal_stdlib>
#include "utils.h"

using namespace metal;

__global__ void reduce_strided(float* g_idata, float* g_odata, unsigned int n) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Reorganized loop: stride halves instead of doubling
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Pack active threads contiguously
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        // Coordinate block-level memory consistency
        __syncthreads();
    }

    // Thread 0 writes out the final sum for this block
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* --- Reduction sum Kernel --- */
template<typename T>
kernel void reduce_sum(
    device const T* src0,
    device T* dst,
    constant ViewInfo& view_src0,
    uint id [[thread_position_in_grid]]
) {
    // create a shared memory
    threadgroup smem[128];

    // get index
    int64_t pid = physical_offset(id, view_src0);

    // load elements into shared memory
    sdata[id] = (id >= view_src0.N) ? 0 : view_src[pid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // reduction loop

    // thread 0 writes out the final sum
    if (id == 0) {
        dst[id] = sdata[0];
    }
}
