/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** | AI assist : Grok 4.5 (Cursor)                                         |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include <cstddef>

#include "comtam/macros/log.h"
#include "comtam/macros/macros.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/view.h"
#include "comtam/types.h"

namespace comtam::checks {

// Matches view_desc / Metal ViewInfo fixed rank encoding.
constexpr size_int kMaxRank = 4;

COMTAM_INLINE void check_supported_dtype(DType dtype) {
    COMTAM_CHECK_AND_THROW(support_dtype(dtype), std::runtime_error, "unsupported dtype");
}

COMTAM_INLINE void check_rank_le(const view& v, size_int max_rank = kMaxRank) {
    COMTAM_CHECK_AND_THROW(static_cast<size_int>(v.dim()) <= max_rank, std::runtime_error,
                           "tensor rank exceeds the maximum supported rank");
}

COMTAM_INLINE void check_positive_extents(const view& v) {
    for (view_int extent : v.shape) {
        COMTAM_CHECK_AND_THROW(extent > 0, std::runtime_error,
                               "tensor dimensions must be positive");
    }
}

COMTAM_INLINE void check_same_dtype(DType a, DType b) {
    COMTAM_CHECK_AND_THROW(a == b, std::runtime_error, "Two operands must have the same dtype");
}

COMTAM_INLINE void check_tensor_gpu_ready(const view& v, DType dtype) {
    check_supported_dtype(dtype);
    check_rank_le(v);
    check_positive_extents(v);
}

COMTAM_INLINE void check_unary(const view& v, DType dtype) {
    check_tensor_gpu_ready(v, dtype);
}

COMTAM_INLINE view_vector check_binary(const view& a, DType a_dtype, const view& b, DType b_dtype) {
    check_same_dtype(a_dtype, b_dtype);
    check_tensor_gpu_ready(a, a_dtype);
    check_tensor_gpu_ready(b, b_dtype);
    // this should throw if not compatible
    return view::broadcast_shape(a, b);
}

COMTAM_INLINE view_vector check_reduce_full(const view& v, DType dtype) {
    check_tensor_gpu_ready(v, dtype);
    COMTAM_CHECK_AND_THROW(v.is_contiguous(), std::runtime_error,
                           "Cannot do operation on non-contiguous inputs");
    return {};
}

COMTAM_INLINE view_vector check_reduce_axis(const view& v, DType dtype, view_int dim,
                                            bool keep_dim) {
    check_tensor_gpu_ready(v, dtype);
    COMTAM_CHECK_AND_THROW(v.is_contiguous(), std::runtime_error,
                           "Cannot do operation on non-contiguous inputs");
    COMTAM_CHECK_AND_THROW(v.dim() > 0, std::runtime_error,
                           "axis reduction is not defined for rank-0 tensors");
    COMTAM_CHECK_AND_THROW(dim >= 0 && dim < v.dim(), std::runtime_error, "dim out of bounds");

    view_vector out_shape = v.shape;
    if (keep_dim) {
        out_shape[static_cast<size_t>(dim)] = 1;
    } else {
        out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(dim));
    }
    return out_shape;
}

COMTAM_INLINE view_vector check_matmul(const view& a, DType a_dtype, const view& b, DType b_dtype) {
    check_same_dtype(a_dtype, b_dtype);
    check_supported_dtype(a_dtype);
    check_positive_extents(a);
    check_positive_extents(b);

    COMTAM_CHECK_AND_THROW(a.dim() == 2 && b.dim() == 2, std::runtime_error,
                           "Right now, matmul only supports 2D array");
    COMTAM_CHECK_AND_THROW(a.shape[1] == b.shape[0], std::runtime_error,
                           "To do matmul, column of a must match with row of b");
    return {a.shape[0], b.shape[1]};
}
}  // namespace comtam::checks
