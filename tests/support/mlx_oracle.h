#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "comtam/core/view.h"
#include "mlx/c/mlx.h"

namespace comtam::tests::mlx_oracle {

using BinaryOp = int (*)(mlx_array*, mlx_array, mlx_array, mlx_stream);

inline void check(int code, const char* what) {
    if (code != 0) {
        throw std::runtime_error(std::string("mlx-c call failed: ") + what);
    }
}

inline std::vector<int> to_mlx_shape(const std::vector<core::ViewInt>& shape) {
    std::vector<int> result;
    result.reserve(shape.size());

    for (auto dim : shape) {
        if (dim < 0 || dim > std::numeric_limits<int>::max()) {
            throw std::runtime_error("mlx-c oracle shape dimension does not fit int");
        }
        result.push_back(static_cast<int>(dim));
    }

    return result;
}

class Stream {
   public:
    Stream() : stream_(mlx_default_cpu_stream_new()) {
        if (!stream_.ctx) {
            throw std::runtime_error("mlx-c failed to create a default CPU stream");
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) { other.stream_ = mlx_stream_new(); }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            reset();
            stream_ = other.stream_;
            other.stream_ = mlx_stream_new();
        }
        return *this;
    }

    ~Stream() { reset(); }

    [[nodiscard]] mlx_stream get() const { return stream_; }

   private:
    void reset() {
        if (stream_.ctx) {
            mlx_stream_free(stream_);
            stream_ = mlx_stream_new();
        }
    }

    mlx_stream stream_;
};

class Array {
   public:
    Array() : array_(mlx_array_new()) {}

    explicit Array(mlx_array array) : array_(array) {
        if (!array_.ctx) {
            throw std::runtime_error("mlx-c produced an empty array");
        }
    }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    Array(Array&& other) noexcept : array_(other.array_) { other.array_ = mlx_array_new(); }

    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            reset();
            array_ = other.array_;
            other.array_ = mlx_array_new();
        }
        return *this;
    }

    ~Array() { reset(); }

    [[nodiscard]] static Array from_float32(const std::vector<float>& data,
                                            const std::vector<core::ViewInt>& shape) {
        auto mlx_shape = to_mlx_shape(shape);
        return Array(mlx_array_new_data(data.data(), mlx_shape.data(),
                                        static_cast<int>(mlx_shape.size()), MLX_FLOAT32));
    }

    [[nodiscard]] mlx_array get() const { return array_; }

    [[nodiscard]] mlx_array* out_ptr() { return &array_; }

   private:
    void reset() {
        if (array_.ctx) {
            mlx_array_free(array_);
            array_ = mlx_array_new();
        }
    }

    mlx_array array_;
};

inline std::vector<float> to_vector_float32(const Array& array, const Stream& stream) {
    Array contiguous;
    check(mlx_contiguous(contiguous.out_ptr(), array.get(), false, stream.get()), "mlx_contiguous");
    check(mlx_array_eval(contiguous.get()), "mlx_array_eval");
    check(mlx_synchronize(stream.get()), "mlx_synchronize");

    const auto* data = mlx_array_data_float32(contiguous.get());
    if (data == nullptr) {
        throw std::runtime_error("mlx-c array data is unavailable after eval");
    }

    auto size = mlx_array_size(contiguous.get());
    return {data, data + size};
}

inline std::vector<float> binary_float32(const std::vector<float>& lhs,
                                         const std::vector<float>& rhs,
                                         const std::vector<core::ViewInt>& shape, BinaryOp op) {
    Stream stream;
    auto a = Array::from_float32(lhs, shape);
    auto b = Array::from_float32(rhs, shape);
    Array result;

    check(op(result.out_ptr(), a.get(), b.get(), stream.get()), "binary op");
    return to_vector_float32(result, stream);
}

inline std::vector<float> transpose_float32(const std::vector<float>& data,
                                            const std::vector<core::ViewInt>& shape,
                                            const std::vector<int>& axes) {
    Stream stream;
    auto input = Array::from_float32(data, shape);
    Array result;

    check(mlx_transpose_axes(result.out_ptr(), input.get(), axes.data(), axes.size(), stream.get()),
          "mlx_transpose_axes");
    return to_vector_float32(result, stream);
}

inline std::vector<float> slice_float32(const std::vector<float>& data,
                                        const std::vector<core::ViewInt>& shape,
                                        const std::vector<int>& start, const std::vector<int>& stop,
                                        const std::vector<int>& strides) {
    Stream stream;
    auto input = Array::from_float32(data, shape);
    Array result;

    check(mlx_slice(result.out_ptr(), input.get(), start.data(), start.size(), stop.data(),
                    stop.size(), strides.data(), strides.size(), stream.get()),
          "mlx_slice");
    return to_vector_float32(result, stream);
}

inline std::vector<float> broadcast_to_float32(const std::vector<float>& data,
                                               const std::vector<core::ViewInt>& shape,
                                               const std::vector<core::ViewInt>& target_shape) {
    Stream stream;
    auto input = Array::from_float32(data, shape);
    auto mlx_shape = to_mlx_shape(target_shape);
    Array result;

    check(mlx_broadcast_to(result.out_ptr(), input.get(), mlx_shape.data(), mlx_shape.size(),
                           stream.get()),
          "mlx_broadcast_to");
    return to_vector_float32(result, stream);
}

inline std::vector<float> reshape_float32(const std::vector<float>& data,
                                          const std::vector<core::ViewInt>& shape,
                                          const std::vector<core::ViewInt>& target_shape) {
    Stream stream;
    auto input = Array::from_float32(data, shape);
    auto mlx_shape = to_mlx_shape(target_shape);
    Array result;

    check(mlx_reshape(result.out_ptr(), input.get(), mlx_shape.data(), mlx_shape.size(),
                      stream.get()),
          "mlx_reshape");
    return to_vector_float32(result, stream);
}

}  // namespace comtam::tests::mlx_oracle
