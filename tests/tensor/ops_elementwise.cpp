/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** | AI assist : Grok 4.5 (Cursor) and GPT 5.5 (Codex)                     |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include <catch2/catch_test_macros.hpp>
#include <cstring>

#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

struct UnaryOpCase {
    mlx_test::UnaryOp mlx_op;
    tensor (*comtam_op)(const tensor&);
};

struct BinaryOpCase {
    mlx_test::BinaryOp mlx_op;
    tensor (*comtam_op)(const tensor&, const tensor&);
};

struct ShapeCase {
    view_vector shape;
    size_t numel;
};

TEST_CASE("Forward compare vs MLX for unary ops", "[forward][ops][mlx][metal]") {
    const UnaryOpCase cases[] = {
        {mlx_test::mx::negative, tensor::neg},
        {mlx_test::mx::reciprocal, tensor::recip},
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
            tensor a(data.data(), shape);

            require_op_matches_oracle(
                a.dtype(), shape, [&]() { return op.comtam_op(a); },
                [&]() { return mlx_test::unary_float32(data, shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}

/*
 * Unary kernels read through physical_offset, so public neg/recip must work on
 * a transposed view and on a shrink with a nonzero storage offset. Contiguous
 * MLX cases alone do not exercise that path.
 */
TEST_CASE("Forward compare vs MLX for unary ops on non-contiguous inputs",
          "[forward][ops][unary][mlx][metal]") {
    const UnaryOpCase cases[] = {
        {mlx_test::mx::negative, tensor::neg},
        {mlx_test::mx::reciprocal, tensor::recip},
    };

    SECTION("transpose") {
        // Logical (4, 3) from a contiguous (3, 4) base: strides become (1, 4).
        const view_vector base_shape{3, 4};
        const view_vector logical_shape{4, 3};
        auto base_data = utils::generate_random_array<float>(12, 1.0F, 2.0F);

        tensor a_base(base_data.data(), base_shape);
        tensor a = a_base.transpose(0, 1);
        REQUIRE(a.shape() == logical_shape);
        REQUIRE(a.strides() == view_vector{1, 4});

        const auto a_equiv = mlx_test::transpose_float32(base_data, base_shape, {1, 0});

        for (const auto& op : cases) {
            CAPTURE(op.mlx_op);
            require_op_matches_oracle(
                a.dtype(), logical_shape, [&]() { return op.comtam_op(a); },
                [&]() { return mlx_test::unary_float32(a_equiv, logical_shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }

    SECTION("nonzero-offset shrink") {
        // Inner 2x2 of a contiguous 3x3 starts at storage offset 4.
        const view_vector base_shape{3, 3};
        const view_vector logical_shape{2, 2};
        auto base_data = utils::generate_random_array<float>(9, 1.0F, 2.0F);

        tensor a_base(base_data.data(), base_shape);
        tensor a = a_base.shrink({{1, 3}, {1, 3}});
        REQUIRE(a.shape() == logical_shape);
        REQUIRE(a.offset() == 4);
        REQUIRE(a.strides() == view_vector{3, 1});

        const auto a_equiv = mlx_test::slice_float32(base_data, base_shape, {1, 1}, {3, 3}, {1, 1});

        for (const auto& op : cases) {
            CAPTURE(op.mlx_op);
            require_op_matches_oracle(
                a.dtype(), logical_shape, [&]() { return op.comtam_op(a); },
                [&]() { return mlx_test::unary_float32(a_equiv, logical_shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}

TEST_CASE("Forward compare vs MLX for elementwise ops", "[forward][ops][mlx][metal]") {
    const BinaryOpCase cases[] = {
        {mlx_test::mx::add, tensor::add},
        {mlx_test::mx::subtract, tensor::sub},
        {mlx_test::mx::multiply, tensor::mul},
        {mlx_test::mx::divide, tensor::div},
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
            tensor a(lhs.data(), shape);
            tensor b(rhs.data(), shape);

            require_op_matches_oracle(
                a.dtype(), shape, [&]() { return op.comtam_op(a, b); },
                [&]() { return mlx_test::binary_float32(lhs, shape, rhs, shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}
