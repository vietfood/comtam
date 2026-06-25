#include <metal_stdlib>

using namespace metal;

kernel void add(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

kernel void sub(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] - inB[index];
}

kernel void mul(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}

kernel void div(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    if (inB[index] == 0.f) {
        result[index] = 0.f;
    } else {
        result[index] = inA[index] / inB[index];
    }
}