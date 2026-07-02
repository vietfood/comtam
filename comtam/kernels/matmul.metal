#include <metal_stdlib>
#include "utils.h"

using namespace metal;

template<typename T>
kernel void matmul(
    device const T* src0,
    device const T* src1,
    device T* dst,
    constant ViewInfo& view_src0,
    constant ViewInfo& view_src1,
    constant uint& BLOCK_SIZE,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]]
) {
    const uint col = gid.x * BLOCK_SIZE + lid.x;
    const uint row = gid.y * BLOCK_SIZE + lid.y;

    // here, we assume the array is 2D only
    const uint M = static_cast<uint>(view_src0.shape[0]);
    const uint K = static_cast<uint>(view_src0.shape[1]);
    const uint N = static_cast<uint>(view_src1.shape[1]);

    if (row < M && col < N) {
        T sum = 0;
        for (uint k = 0; k < K; ++k) {
            // get linear index
            uint src0_id = row * K + k;
            uint src1_id = k * N + col;

            // get physical index
            uint src0_pid = physical_offset(src0_id, view_src0);
            uint src1_pid = physical_offset(src1_id, view_src1);

            // calculate sum
            sum += src0[src0_pid] * src1[src1_pid];
        }
        dst[row * N + col] = sum;
    }
}

template [[ host_name("matmul_fp32") ]]
kernel void matmul(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    constant uint& BLOCK_SIZE,
    uint3,
    uint3
);
