#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/dtype.h"
#include "comtam/core/view.h"
#include "comtam/tensor.h"

namespace comtam::tests::forward_compare {

constexpr double kDefaultEpsilon = 1e-5;

enum class ValueMode {
    Exact,
    Approximate,
};

inline size_t numel_from_shape(const std::vector<core::ViewInt>& shape) {
    return static_cast<size_t>(
        std::reduce(shape.begin(), shape.end(), core::ViewInt{1}, std::multiplies<>()));
}

inline void require_shape_matches(const Tensor& actual,
                                  const std::vector<core::ViewInt>& expected_shape) {
    CAPTURE(expected_shape);
    REQUIRE(actual.dim() == expected_shape.size());
    REQUIRE(actual.numel() == numel_from_shape(expected_shape));
    REQUIRE(actual.shape() == expected_shape);
}

template <typename T>
inline void require_values_exact(const std::vector<T>& expected, const std::vector<T>& actual) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE(actual[i] == expected[i]);
    }
}

template <typename T>
inline void require_values_close(const std::vector<T>& expected, const std::vector<T>& actual,
                                 double epsilon = kDefaultEpsilon) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(static_cast<double>(actual[i]),
                     Catch::Matchers::WithinAbs(static_cast<double>(expected[i]), epsilon));
    }
}

template <typename T>
inline void require_forward_matches(core::Device& device, const Tensor& actual,
                                    const std::vector<core::ViewInt>& expected_shape,
                                    const std::vector<T>& expected_values, ValueMode mode,
                                    float epsilon = kDefaultEpsilon) {
    require_shape_matches(actual, expected_shape);
    REQUIRE(expected_values.size() == numel_from_shape(expected_shape));

    auto actual_values = actual.to_vector<T>(device);
    if (mode == ValueMode::Exact) {
        require_values_exact<T>(expected_values, actual_values);
    } else {
        require_values_close<T>(expected_values, actual_values, epsilon);
    }
}

template <typename ComtamFn, typename OracleFn>
void require_op_matches_oracle(core::Context& context, const core::DType& dtype,
                               const std::vector<core::ViewInt>& expected_shape,
                               ComtamFn&& comtam_fn, OracleFn&& oracle_fn, ValueMode mode,
                               float epsilon = kDefaultEpsilon) {
    COMTAM_DISPATCH_DTYPE(dtype, [&] {
        auto& device = context.device();
        require_forward_matches<scalar_t>(device, comtam_fn(), expected_shape, oracle_fn(), mode, epsilon);
    });
}

}  // namespace comtam::tests::forward_compare
