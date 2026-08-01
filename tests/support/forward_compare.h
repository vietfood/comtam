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

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/view.h"
#include "comtam/tensor/tensor.h"

namespace comtam::tests::forward_compare {

constexpr double kDefaultEpsilon = 1e-5;
constexpr size_t kMaxDisplayedElements = 64;
constexpr size_t kMismatchWindowRadius = 4;

enum class ValueMode {
    Exact,
    Approximate,
};

inline size_t numel_from_shape(const view_vector& shape) {
    return static_cast<size_t>(
        std::reduce(shape.begin(), shape.end(), view_int{1}, std::multiplies<>()));
}

template <typename T>
inline void append_shaped_values(std::string& out, const std::vector<T>& values,
                                 const view_vector& shape, size_t& index, size_t dim,
                                 const std::string& indent) {
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
inline std::string format_values(const std::vector<T>& values, const view_vector& shape = {}) {
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
                                         const view_vector& shape = {},
                                         size_t max_elements = kMaxDisplayedElements) {
    if (values.size() <= max_elements) {
        return format_values(values, shape);
    }

    const std::vector<T> prefix(values.begin(), values.begin() + max_elements);
    if (shape.empty()) {
        return fmt::format("{} ... ({} elements total)", fmt::format("{}", prefix), values.size());
    }
    return fmt::format("{} ... ({} elements total, shape={})", format_values(prefix, {}),
                       values.size(), fmt::format("{}", shape));
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

inline void require_shape_matches(const tensor& actual, const view_vector& expected_shape) {
    CAPTURE(expected_shape);
    REQUIRE(actual.dim() == expected_shape.size());
    REQUIRE(actual.numel() == numel_from_shape(expected_shape));
    REQUIRE(actual.shape() == expected_shape);
}

template <typename T>
inline void require_values_exact(const std::vector<T>& expected, const std::vector<T>& actual,
                                 const view_vector& shape = {}) {
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
                                 double epsilon = kDefaultEpsilon, const view_vector& shape = {}) {
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
inline void require_forward_matches(core::metal_device& device, const tensor& actual,
                                    const view_vector& expected_shape,
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
void require_op_matches_oracle(core::context& context, const DType& dtype,
                               const view_vector& expected_shape, ComtamFn&& comtam_fn,
                               OracleFn&& oracle_fn, ValueMode mode,
                               float epsilon = kDefaultEpsilon) {
    COMTAM_DISPATCH_DTYPE(dtype, [&] {
        auto& device = context.device();
        require_forward_matches<scalar_t>(device, comtam_fn(), expected_shape, oracle_fn(), mode,
                                          epsilon);
    });
}

enum class FloatClass {
    Finite,
    PosZero,
    NegZero,
    PosInf,
    NegInf,
    NaN,
};

inline FloatClass classify_float32(float value) {
    if (std::isnan(value)) {
        return FloatClass::NaN;
    }
    if (std::isinf(value)) {
        return value > 0 ? FloatClass::PosInf : FloatClass::NegInf;
    }
    if (value == 0.0F) {
        return std::signbit(value) ? FloatClass::NegZero : FloatClass::PosZero;
    }
    return FloatClass::Finite;
}

inline const char* float_class_name(FloatClass klass) {
    switch (klass) {
        case FloatClass::Finite:
            return "finite";
        case FloatClass::PosZero:
            return "+0";
        case FloatClass::NegZero:
            return "-0";
        case FloatClass::PosInf:
            return "+inf";
        case FloatClass::NegInf:
            return "-inf";
        case FloatClass::NaN:
            return "nan";
    }
    return "unknown";
}

inline void require_float_class(float actual, FloatClass expected) {
    const auto actual_class = classify_float32(actual);
    CAPTURE(actual);
    CAPTURE(float_class_name(actual_class));
    CAPTURE(float_class_name(expected));
    REQUIRE(actual_class == expected);
}

inline void require_values_close_abs_rel(const std::vector<float>& expected,
                                         const std::vector<float>& actual, double abs_eps,
                                         double rel_eps, const view_vector& shape = {}) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const bool close =
            Catch::Matchers::WithinAbs(static_cast<double>(expected[i]), abs_eps)
                .match(static_cast<double>(actual[i])) ||
            Catch::Matchers::WithinRel(static_cast<double>(expected[i]), rel_eps)
                .match(static_cast<double>(actual[i]));
        if (close) {
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
        REQUIRE(close);
    }
}

}  // namespace comtam::tests::forward_compare
