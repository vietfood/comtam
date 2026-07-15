#include <metal_stdlib>
#include "utils.h"

using namespace metal;

// reference from CUDA
// will delete later
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
    uint2 block_pos [[ threadgroup_position_in_grid ]],
    uint2 thread_pos [[ thread_position_in_threadgroup ]]
) {
}
