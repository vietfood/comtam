/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** | AI assist : Composer 2.5 (Cursor) and GPT 5.5 (Codex)                 |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cstddef>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <numeric>
#include <string>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/dtype.h"
#include "comtam/core/view.h"
#include "comtam/tensor.h"

namespace comtam::tests::forward_compare {

constexpr double kDefaultEpsilon = 1e-5;
constexpr size_t kMaxDisplayedElements = 64;
constexpr size_t kMismatchWindowRadius = 4;

enum class ValueMode {
    Exact,
    Approximate,
};

inline size_t numel_from_shape(const std::vector<core::ViewInt>& shape) {
    return static_cast<size_t>(
        std::reduce(shape.begin(), shape.end(), core::ViewInt{1}, std::multiplies<>()));
}

template <typename T>
inline void append_shaped_values(std::string& out, const std::vector<T>& values,
                                 const std::vector<core::ViewInt>& shape, size_t& index,
                                 size_t dim, const std::string& indent) {
    out += "[";
    const auto extent = static_cast<size_t>(shape[dim]);
    for (size_t i = 0; i < extent; ++i) {
        if (i > 0) {
            out += ",\n" + indent;
        }
        if (dim + 1 == shape.size()) {
            out += fmt::format("{}", values[index++]);
        } else {
            append_shaped_values(out, values, shape, index, dim + 1, indent + "  ");
        }
    }
    out += "]";
}

template <typename T>
inline std::string format_values(const std::vector<T>& values,
                                 const std::vector<core::ViewInt>& shape = {}) {
    if (values.empty()) {
        return "[]";
    }
    if (shape.empty()) {
        return fmt::format("{}", values);
    }

    size_t index = 0;
    std::string out;
    append_shaped_values(out, values, shape, index, 0, "");
    return out;
}

template <typename T>
inline std::string format_values_preview(const std::vector<T>& values,
                                         const std::vector<core::ViewInt>& shape = {},
                                         size_t max_elements = kMaxDisplayedElements) {
    if (values.size() <= max_elements) {
        return format_values(values, shape);
    }

    const std::vector<T> prefix(values.begin(), values.begin() + max_elements);
    if (shape.empty()) {
        return fmt::format("{} ... ({} elements total)", fmt::format("{}", prefix), values.size());
    }
    return fmt::format("{} ... ({} elements total, shape={})", format_values(prefix, {}), values.size(),
                       fmt::format("{}", shape));
}

template <typename T>
inline std::string format_mismatch_window(const std::vector<T>& expected,
                                          const std::vector<T>& actual, size_t index,
                                          size_t radius = kMismatchWindowRadius) {
    const size_t start = index > radius ? index - radius : 0;
    const size_t end = std::min(expected.size(), index + radius + 1);
    std::vector<T> expected_window(expected.begin() + static_cast<std::ptrdiff_t>(start),
                                   expected.begin() + static_cast<std::ptrdiff_t>(end));
    std::vector<T> actual_window(actual.begin() + static_cast<std::ptrdiff_t>(start),
                                 actual.begin() + static_cast<std::ptrdiff_t>(end));
    return fmt::format("indices [{}..{}): expected={}, actual={}", start, end,
                       fmt::format("{}", expected_window), fmt::format("{}", actual_window));
}

inline void require_shape_matches(const Tensor& actual,
                                  const std::vector<core::ViewInt>& expected_shape) {
    CAPTURE(expected_shape);
    REQUIRE(actual.dim() == expected_shape.size());
    REQUIRE(actual.numel() == numel_from_shape(expected_shape));
    REQUIRE(actual.shape() == expected_shape);
}

template <typename T>
inline void require_values_exact(const std::vector<T>& expected, const std::vector<T>& actual,
                                 const std::vector<core::ViewInt>& shape = {}) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] == expected[i]) {
            continue;
        }

        const std::string expected_preview = format_values_preview(expected, shape);
        const std::string actual_preview = format_values_preview(actual, shape);
        const std::string mismatch_window = format_mismatch_window(expected, actual, i);
        CAPTURE(expected_preview);
        CAPTURE(actual_preview);
        CAPTURE(mismatch_window);
        CAPTURE(i);
        CAPTURE(expected[i]);
        CAPTURE(actual[i]);
        REQUIRE(actual[i] == expected[i]);
    }
}

template <typename T>
inline void require_values_close(const std::vector<T>& expected, const std::vector<T>& actual,
                                 double epsilon = kDefaultEpsilon,
                                 const std::vector<core::ViewInt>& shape = {}) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (Catch::Matchers::WithinAbs(static_cast<double>(expected[i]), epsilon)
                .match(static_cast<double>(actual[i]))) {
            continue;
        }

        const std::string expected_preview = format_values_preview(expected, shape);
        const std::string actual_preview = format_values_preview(actual, shape);
        const std::string mismatch_window = format_mismatch_window(expected, actual, i);
        CAPTURE(expected_preview);
        CAPTURE(actual_preview);
        CAPTURE(mismatch_window);
        CAPTURE(i);
        CAPTURE(expected[i]);
        CAPTURE(actual[i]);
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
        require_values_exact<T>(expected_values, actual_values, expected_shape);
    } else {
        require_values_close<T>(expected_values, actual_values, epsilon, expected_shape);
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
