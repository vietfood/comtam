#pragma once

#include <metal_stdlib>
using namespace metal;

/*
 * We assume that shape and stride have at most 4 elements.
 *
 * Layout must match core/command.h view_desc exactly: the host sends view_desc
 * raw via setBytes and we reinterpret it as `constant ViewInfo&`.
 */
struct ViewInfo {
    size_t N;
    int64_t shape[4];
    int64_t strides[4];
    int64_t offset;
    bool contiguous;
};

/*
 * linear_index is the Metal thread position (uint, 32-bit), so this kernel
 * caps total elements at 2^32 per dispatch. shape/strides/offset stay 64-bit
 * so physical offsets can exceed 2^32 without overflow.
 */
inline int64_t physical_offset(
    uint linear_index,
    constant ViewInfo& view
) {
    if (view.contiguous) {
        return static_cast<int64_t>(linear_index) + view.offset;
    }

    int64_t offset = view.offset;

    /* calculation for non-contiguous view */
    int64_t tmp = static_cast<int64_t>(linear_index);
    int64_t dec[4] {0};

    for (int64_t i = 3; i >= 0; --i) {
        if (view.shape[i] == -1) {
            /* -1 marks an undefined shape slot, so skip it */
            dec[i] = -1;
            continue;
        }
        dec[i] = tmp % view.shape[i];
        tmp /= view.shape[i];
    }

    int64_t res = offset;
    for (size_t i = 0; i < 4; ++i) {
        if (dec[i] == -1) {
            continue;
        }
        res += dec[i] * view.strides[i];
    }
    return res;
}
