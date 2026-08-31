/*
** +--( ~_~ )-----------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                          |
** | Licensed under the Apache License, Version 2.0                           |
** | AI assist : Composer 2.5 (Cursor), Grok 4.5 (Cursor) and GPT 5.5 (Codex) |
** |                                                                          |
** | Website : https://lenguyen.vercel.app                                    |
** | GitHub  : https://github.com/vietfood/comtam                             |
** | License : https://www.apache.org/licenses/LICENSE-2.0                    |
** +--( ^_^ )-----------------------------------------------------------------+
*/

#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include "comtam/tensor/view.h"
#include "mlx/mlx.h"

namespace comtam::tests::mlx_oracle {

namespace mx = mlx::core;

using BinaryOp = mx::array (*)(const mx::array&, const mx::array&, mx::StreamOrDevice);
using UnaryOp = mx::array (*)(const mx::array&, mx::StreamOrDevice);

inline mx::Stream cpu_stream() { return mx::default_stream(mx::Device::cpu); }

inline mx::Shape to_mlx_shape(const view_vector& shape) {
    mx::Shape result;
    result.reserve(shape.size());

    for (auto dim : shape) {
        if (dim < 0 || dim > std::numeric_limits<mx::ShapeElem>::max()) {
            throw std::runtime_error("mlx oracle shape dimension does not fit ShapeElem");
        }
        result.push_back(static_cast<mx::ShapeElem>(dim));
    }

    return result;
}

inline mx::Shape to_mlx_shape(const std::vector<int>& values) {
    return mx::Shape(values.begin(), values.end());
}

inline mx::array from_float32(const std::vector<float>& data, const view_vector& shape) {
    return mx::array(data.data(), to_mlx_shape(shape), mx::float32);
}

inline std::vector<float> to_vector_float32(const mx::array& array, mx::Stream stream) {
    auto contiguous = mx::contiguous(array, false, stream);
    contiguous.eval();
    mx::synchronize(stream);

    const auto* data = contiguous.data<float>();
    if (data == nullptr) {
        throw std::runtime_error("mlx array data is unavailable after eval");
    }

    return {data, data + contiguous.size()};
}

inline std::vector<float> unary_float32(const std::vector<float>& lhs, const view_vector& shape,
                                        UnaryOp op) {
    auto stream = cpu_stream();
    return to_vector_float32(op(from_float32(lhs, shape), stream), stream);
}

inline std::vector<float> binary_float32(const std::vector<float>& lhs,
                                         const view_vector& lhs_shape,
                                         const std::vector<float>& rhs,
                                         const view_vector& rhs_shape, BinaryOp op) {
    auto stream = cpu_stream();
    return to_vector_float32(
        op(from_float32(lhs, lhs_shape), from_float32(rhs, rhs_shape), stream), stream);
}

inline std::vector<float> matmul_float32(const std::vector<float>& lhs,
                                         const view_vector& lhs_shape,
                                         const std::vector<float>& rhs,
                                         const view_vector& rhs_shape) {
    auto stream = cpu_stream();
    return to_vector_float32(
        mx::matmul(from_float32(lhs, lhs_shape), from_float32(rhs, rhs_shape), stream), stream);
}

inline std::vector<float> transpose_float32(const std::vector<float>& data,
                                            const view_vector& shape,
                                            const std::vector<int>& axes) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::transpose(from_float32(data, shape), axes, stream), stream);
}

inline std::vector<float> slice_float32(const std::vector<float>& data, const view_vector& shape,
                                        const std::vector<int>& start, const std::vector<int>& stop,
                                        const std::vector<int>& strides) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::slice(from_float32(data, shape), to_mlx_shape(start),
                                       to_mlx_shape(stop), to_mlx_shape(strides), stream),
                             stream);
}

inline std::vector<float> broadcast_to_float32(const std::vector<float>& data,
                                               const view_vector& shape,
                                               const view_vector& target_shape) {
    auto stream = cpu_stream();
    return to_vector_float32(
        mx::broadcast_to(from_float32(data, shape), to_mlx_shape(target_shape), stream), stream);
}

inline std::vector<float> reshape_float32(const std::vector<float>& data, const view_vector& shape,
                                          const view_vector& target_shape) {
    auto stream = cpu_stream();
    return to_vector_float32(
        mx::reshape(from_float32(data, shape), to_mlx_shape(target_shape), stream), stream);
}

inline std::vector<float> sum_float32(const std::vector<float>& data, const view_vector& shape,
                                      bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::sum(from_float32(data, shape), keepdims, stream), stream);
}

inline std::vector<float> sum_axis_float32(const std::vector<float>& data, const view_vector& shape,
                                           int axis, bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::sum(from_float32(data, shape), axis, keepdims, stream), stream);
}

inline std::vector<float> mean_float32(const std::vector<float>& data, const view_vector& shape,
                                       bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::mean(from_float32(data, shape), keepdims, stream), stream);
}

inline std::vector<float> mean_axis_float32(const std::vector<float>& data,
                                            const view_vector& shape, int axis,
                                            bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::mean(from_float32(data, shape), axis, keepdims, stream), stream);
}

inline std::vector<float> max_float32(const std::vector<float>& data, const view_vector& shape,
                                      bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::max(from_float32(data, shape), keepdims, stream), stream);
}

inline std::vector<float> max_axis_float32(const std::vector<float>& data, const view_vector& shape,
                                           int axis, bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::max(from_float32(data, shape), axis, keepdims, stream), stream);
}

inline std::vector<float> min_float32(const std::vector<float>& data, const view_vector& shape,
                                      bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::min(from_float32(data, shape), keepdims, stream), stream);
}

inline std::vector<float> min_axis_float32(const std::vector<float>& data, const view_vector& shape,
                                           int axis, bool keepdims = false) {
    auto stream = cpu_stream();
    return to_vector_float32(mx::min(from_float32(data, shape), axis, keepdims, stream), stream);
}

}  // namespace comtam::tests::mlx_oracle
