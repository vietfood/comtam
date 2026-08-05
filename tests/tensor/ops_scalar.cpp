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

#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "mlx/c/ops.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::numel_from_shape;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare vs MLX for scalar binary ops", "[forward][ops][scalar][mlx][metal]") {
    struct ShapeCase {
        view_vector shape;
    };

    const ShapeCase shape_cases[] = {
        {{2, 3}},
        {{3, 1}},
        {{4}},
        {},
    };

    const float scalars[] = {2.0F, -0.5F, 1.0F, 0.25F};

    for (const auto& shape_case : shape_cases) {
        for (float scalar : scalars) {
            CAPTURE(shape_case.shape);
            CAPTURE(scalar);

            const auto n = numel_from_shape(shape_case.shape);
            auto lhs = utils::generate_random_array<float>(n, 1.0F, 2.0F);
            tensor a(lhs.data(), shape_case.shape);
            const std::vector<float> rhs{scalar};
            const view_vector scalar_shape{};

            require_op_matches_oracle(
                a.dtype(), shape_case.shape,
                [&]() { return tensor::add(a, tensor(scalar, a.dtype())); },
                [&]() {
                    return mlx_test::binary_float32(lhs, shape_case.shape, rhs, scalar_shape,
                                                    mlx_add);
                },
                ValueMode::Approximate);

            require_op_matches_oracle(
                a.dtype(), shape_case.shape,
                [&]() { return tensor::mul(a, tensor(scalar, a.dtype())); },
                [&]() {
                    return mlx_test::binary_float32(lhs, shape_case.shape, rhs, scalar_shape,
                                                    mlx_multiply);
                },
                ValueMode::Approximate);

            require_op_matches_oracle(
                a.dtype(), shape_case.shape,
                [&]() { return tensor::sub(a, tensor(scalar, a.dtype())); },
                [&]() {
                    return mlx_test::binary_float32(lhs, shape_case.shape, rhs, scalar_shape,
                                                    mlx_subtract);
                },
                ValueMode::Approximate);

            require_op_matches_oracle(
                a.dtype(), shape_case.shape,
                [&]() { return tensor::div(a, tensor(scalar, a.dtype())); },
                [&]() {
                    return mlx_test::binary_float32(lhs, shape_case.shape, rhs, scalar_shape,
                                                    mlx_divide);
                },
                ValueMode::Approximate);
        }
    }
}

TEST_CASE("Scalar binary ops reject mismatched dtype and divide-by-zero",
          "[forward][ops][scalar][metal]") {
    const float data[] = {1.0F, 2.0F, 3.0F, 4.0F};
    tensor a(data, {2, 2});

    REQUIRE_THROWS_AS(tensor::add(a, tensor(1, a.dtype())), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(a, tensor(2UL, a.dtype())), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(a, tensor(0.0, a.dtype())), std::runtime_error);
}
