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

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "comtam/core/context.h"
#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::numel_from_shape;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

// Reductions change summation order vs MLX; allow a slightly looser abs epsilon.
constexpr float kReduceEpsilon = 1e-4F;

view_vector axis_out_shape(const view_vector& shape, view_int axis, bool keep_dim) {
    view_vector out = shape;
    if (keep_dim) {
        out[static_cast<size_t>(axis)] = 1;
    } else {
        out.erase(out.begin() + static_cast<std::ptrdiff_t>(axis));
    }
    return out;
}

TEST_CASE("Forward compare vs MLX for full reductions", "[forward][ops][reduce][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    const view_vector shapes[] = {
        {4, 5},
        {3, 4},
        {2, 3, 4},
        {8},
        {},
    };

    for (const auto& shape : shapes) {
        CAPTURE(shape);

        const auto n = numel_from_shape(shape);
        auto data = utils::generate_random_array<float>(n, 1.0F, 2.0F);
        tensor a(data.data(), shape, device);

        require_op_matches_oracle(
            context, a.dtype(), {}, [&]() { return tensor::sum(a, context); },
            [&]() { return mlx_test::sum_float32(data, shape, false); }, ValueMode::Approximate,
            kReduceEpsilon);

        require_op_matches_oracle(
            context, a.dtype(), {}, [&]() { return tensor::mean(a, context); },
            [&]() { return mlx_test::mean_float32(data, shape, false); }, ValueMode::Approximate,
            kReduceEpsilon);

        require_op_matches_oracle(
            context, a.dtype(), {}, [&]() { return tensor::max(a, context); },
            [&]() { return mlx_test::max_float32(data, shape, false); }, ValueMode::Approximate,
            kReduceEpsilon);

        require_op_matches_oracle(
            context, a.dtype(), {}, [&]() { return tensor::min(a, context); },
            [&]() { return mlx_test::min_float32(data, shape, false); }, ValueMode::Approximate,
            kReduceEpsilon);
    }
}

TEST_CASE("Forward compare vs MLX for axis: sum max min mean", "[forward][ops][reduce][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    const view_vector shape{4, 5};
    const auto n = numel_from_shape(shape);
    auto data = utils::generate_random_array<float>(n, 1.0F, 2.0F);
    tensor a(data.data(), shape, device);

    for (view_int axis : {0, 1}) {
        for (bool keep_dim : {false, true}) {
            CAPTURE(axis);
            CAPTURE(keep_dim);

            const auto expected_shape = axis_out_shape(shape, axis, keep_dim);

            require_op_matches_oracle(
                context, a.dtype(), expected_shape,
                [&]() { return tensor::sum(a, axis, keep_dim, context); },
                [&]() {
                    return mlx_test::sum_axis_float32(data, shape, static_cast<int>(axis),
                                                      keep_dim);
                },
                ValueMode::Approximate, kReduceEpsilon);

            require_op_matches_oracle(
                context, a.dtype(), expected_shape,
                [&]() { return tensor::mean(a, axis, keep_dim, context); },
                [&]() {
                    return mlx_test::mean_axis_float32(data, shape, static_cast<int>(axis),
                                                      keep_dim);
                },
                ValueMode::Approximate, kReduceEpsilon);

            require_op_matches_oracle(
                context, a.dtype(), expected_shape,
                [&]() { return tensor::max(a, axis, keep_dim, context); },
                [&]() {
                    return mlx_test::max_axis_float32(data, shape, static_cast<int>(axis),
                                                      keep_dim);
                },
                ValueMode::Approximate, kReduceEpsilon);

            require_op_matches_oracle(
                context, a.dtype(), expected_shape,
                [&]() { return tensor::min(a, axis, keep_dim, context); },
                [&]() {
                    return mlx_test::min_axis_float32(data, shape, static_cast<int>(axis),
                                                      keep_dim);
                },
                ValueMode::Approximate, kReduceEpsilon);
        }
    }
}

TEST_CASE("Axis sum and max reject invalid axes", "[forward][ops][reduce][metal]") {
    core::context context;
    auto& device = context.device();

    const float data[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    tensor a(data, {2, 3}, device);
    tensor scalar(data, {}, device);

    REQUIRE_THROWS_AS(tensor::sum(a, -1, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sum(a, 2, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::max(a, 5, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::min(a, 5, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(a, 5, false, context), std::runtime_error);

    // Rank-0 has no valid axis.
    REQUIRE_THROWS_AS(tensor::sum(scalar, 0, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::max(scalar, 0, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(scalar, 0, false, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::min(scalar, 0, false, context), std::runtime_error);
}
