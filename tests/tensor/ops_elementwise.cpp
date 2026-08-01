/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** | AI assist : Grok 4.5 (Cursor) and GPT 5.5 (Codex)                     |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include <catch2/catch_test_macros.hpp>
#include <cstring>

#include "comtam/core/context.h"
#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "mlx/c/ops.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

struct UnaryOpCase {
    mlx_test::UnaryOp mlx_op;
    tensor (*comtam_op)(const tensor&, core::context&);
};

struct BinaryOpCase {
    mlx_test::BinaryOp mlx_op;
    tensor (*comtam_op)(const tensor&, const tensor&, core::context&);
};

struct ShapeCase {
    view_vector shape;
    size_t numel;
};


TEST_CASE("Forward compare vs MLX for unary ops", "[forward][ops][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    const UnaryOpCase cases[] = {
        {mlx_negative, tensor::neg},
        {mlx_reciprocal, tensor::recip},
    };

    const ShapeCase shape_cases[] = {
        {{2, 3}, 6}, {{3, 1}, 3}, {{1, 4}, 4}, {{3, 4}, 12}, {{2, 3, 4}, 24},
    };

    for (const auto& shape_case : shape_cases) {
        for (const auto& op : cases) {
            CAPTURE(op.mlx_op);
            CAPTURE(shape_case.shape);

            auto shape = shape_case.shape;

            auto data = utils::generate_random_array<float>(shape_case.numel, 1.0F, 2.0F);
            tensor a(data.data(), shape, device);

            require_op_matches_oracle(
                context, a.dtype(), shape, [&]() { return op.comtam_op(a, context); },
                [&]() { return mlx_test::unary_float32(data, shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}

TEST_CASE("Forward compare vs MLX for elementwise ops", "[forward][ops][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    const BinaryOpCase cases[] = {
        {mlx_add, tensor::add},
        {mlx_subtract, tensor::sub},
        {mlx_multiply, tensor::mul},
        {mlx_divide, tensor::div},
    };

    const ShapeCase shape_cases[] = {
        {{2, 3}, 6}, {{3, 1}, 3}, {{1, 4}, 4}, {{3, 4}, 12}, {{2, 3, 4}, 24}};

    for (const auto& op : cases) {
        for (const auto& shape_case : shape_cases) {
            CAPTURE(op.mlx_op);
            CAPTURE(shape_case.shape);

            auto shape = shape_case.shape;

            auto lhs = utils::generate_random_array<float>(shape_case.numel, 1.0F, 2.0F);
            auto rhs = utils::generate_random_array<float>(shape_case.numel, 0.5F, 1.5F);
            tensor a(lhs.data(), shape, device);
            tensor b(rhs.data(), shape, device);

            require_op_matches_oracle(
                context, a.dtype(), shape, [&]() { return op.comtam_op(a, b, context); },
                [&]() { return mlx_test::binary_float32(lhs, shape, rhs, shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}
