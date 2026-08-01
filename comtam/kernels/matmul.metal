#include <metal_stdlib>
#include "utils.h"

using namespace metal;

/*
 * This is implementation of matmul tiling with 16x16 block size
 */
template <typename T>
kernel void matmul_contiguous(
    device const T* src0,
    device const T* src1,
    device T* dst,
    constant ViewInfo& view_src0,
    constant ViewInfo& view_src1,
    constant uint& BLOCK_SIZE,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]]
) {
    assert(BLOCK_SIZE == 16);

    // here, we assume the array is 2D only
    const uint M = static_cast<uint>(view_src0.shape[0]);
    const uint K = static_cast<uint>(view_src0.shape[1]);
    const uint N = static_cast<uint>(view_src1.shape[1]);

    // get current column and row of thread
    const uint b_row = gid.y, b_col = gid.x;
    const uint t_row = lid.y, t_col = lid.x;

    /*
    * Instead of traversing using a global index
    * we advance pointer to current row (for A)
    * or current column (for B)
    * and then compute tile from there
    */
    src0 += b_row * BLOCK_SIZE * K;
    src1 += b_col * BLOCK_SIZE;
    dst += b_row * BLOCK_SIZE * N + b_col * BLOCK_SIZE;

    float sum = 0.f;
    threadgroup float smem0[16 * 16];
    threadgroup float smem1[16 * 16];

    for (uint ph = 0; ph < K; ph += BLOCK_SIZE) {
        // load data to shared memory

        // load src0
        if (t_row + b_row * BLOCK_SIZE < M && ph + t_col < K) {
            smem0[t_row * BLOCK_SIZE + t_col] = src0[t_row * K + t_col];
        } else {
            smem0[t_row * BLOCK_SIZE + t_col] = 0.f;
        }

        // load src1
        if (ph + t_row < K && t_col + b_col * BLOCK_SIZE < N) {
            smem1[t_row * BLOCK_SIZE + t_col] = src1[t_row * N + t_col];
        } else {
            smem1[t_row * BLOCK_SIZE + t_col] = 0.f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BLOCK_SIZE; k++) {
            sum += smem0[t_row * BLOCK_SIZE + k] * smem1[k * BLOCK_SIZE + t_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        src0 += BLOCK_SIZE;
        src1 += BLOCK_SIZE * N;
    }

    if (t_row + b_row * BLOCK_SIZE < M && t_col + b_col * BLOCK_SIZE < N) {
        dst[t_row * N + t_col] = sum;
    }
}

template <typename T>
kernel void matmul_strided(
    device const T* src0,
    device const T* src1,
    device T* dst,
    constant ViewInfo& view_src0,
    constant ViewInfo& view_src1,
    constant uint& BLOCK_SIZE,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]]
) {
    // here, we assume the array is 2D only
    const uint M = static_cast<uint>(view_src0.shape[0]);
    const uint K = static_cast<uint>(view_src0.shape[1]);
    const uint N = static_cast<uint>(view_src1.shape[1]);

    const uint col = gid.x * BLOCK_SIZE + lid.x;
    const uint row = gid.y * BLOCK_SIZE + lid.y;
    if (row < M && col < N) {
        T sum = 0;
        for (uint k = 0; k < K; ++k) {
            // get linear index
            uint src0_id = row * K + k;
            uint src1_id = k * N + col;

            // get physical index
            int64_t src0_pid = physical_offset(src0_id, view_src0);
            int64_t src1_pid = physical_offset(src1_id, view_src1);

            // calculate sum
            sum += src0[src0_pid] * src1[src1_pid];
        }
        dst[row * N + col] = sum;
    }
}

template [[ host_name("matmul_contiguous_fp32") ]]
kernel void matmul_contiguous(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    constant uint& BLOCK_SIZE,
    uint3,
    uint3
);

template [[ host_name("matmul_strided_fp32") ]]
kernel void matmul_strided(
    device const float*,
    device const float*,
    device float*,
    constant ViewInfo&,
    constant ViewInfo&,
    constant uint& BLOCK_SIZE,
    uint3,
    uint3
);
