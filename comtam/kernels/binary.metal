#include <metal_stdlib>

using namespace metal;

/* --- Add Kernel --- */
// https://stackoverflow.com/questions/56687496/how-to-make-templated-compute-kernels-in-metal
template<typename T>
kernel void add(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant size_t& N,
        uint id [[thread_position_in_grid]]) {
    if (id >= N) {
        return;
    }
    dst[id] = src0[id] + src1[id];
}

template [[ host_name("add_fp32") ]]
kernel void add(device const float*, device const float*, device float*, constant size_t&, uint);

/* --- Sub Kernel --- */
template<typename T>
kernel void sub(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant size_t& N,
        uint id [[thread_position_in_grid]]) {
    if (id >= N) {
        return;
    }
    dst[id] = src0[id] - src1[id];
}

template [[ host_name("sub_fp32") ]]
kernel void sub(device const float*, device const float*, device float*, constant size_t&, uint);

/* -- Mul Kernel --- */
template<typename T>
kernel void mul(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant size_t& N,
        uint id [[thread_position_in_grid]]) {
    if (id >= N) {
        return;
    }
    dst[id] = src0[id] * src1[id];
}

template [[ host_name("mul_fp32") ]]
kernel void mul(device const float*, device const float*, device float*, constant size_t&, uint);

/* -- Div Kernel --- */
template<typename T>
kernel void div(
        device const T* src0,
        device const T* src1,
        device T* dst,
        constant size_t& N,
        uint id [[thread_position_in_grid]]) {
    if (id >= N) {
        return;
    }

    if (src1[id] == 0) {
        dst[id] = INFINITY;
    } else {
        dst[id] = src0[id] / src1[id];
    }
}

template [[ host_name("div_fp32") ]]
kernel void div(device const float*, device const float*, device float*, constant size_t&, uint);
