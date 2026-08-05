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

#include "comtam/tensor/tensor.h"
#include "comtam/tensor/view.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::numel_from_shape;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare vs MLX for broadcast binary ops",
          "[forward][ops][broadcast][mlx][metal]") {
    struct OpCase {
        mlx_test::BinaryOp mlx_op;
        tensor (*comtam_op)(const tensor&, const tensor&);
    };

    const OpCase op_cases[] = {
        {mlx_add, tensor::add},
        {mlx_subtract, tensor::sub},
        {mlx_multiply, tensor::mul},
        {mlx_divide, tensor::div},
    };

    struct BroadcastCase {
        view_vector lhs_shape;
        view_vector rhs_shape;
    };

    const BroadcastCase broadcast_cases[] = {
        {{4, 3}, {3}},    {{3}, {4, 3}},    {{2, 1, 4}, {3, 1}}, {{3, 1}, {2, 1, 4}},
        {{3, 4}, {3, 1}}, {{3, 4}, {1, 4}}, {{4, 1}, {1, 3}},    {{1}, {3, 4}},
    };

    for (const auto& op : op_cases) {
        for (const auto& broadcast_case : broadcast_cases) {
            CAPTURE(op.mlx_op);
            CAPTURE(broadcast_case.lhs_shape);
            CAPTURE(broadcast_case.rhs_shape);

            const auto expected_shape = view::broadcast_shape(view(broadcast_case.lhs_shape),
                                                              view(broadcast_case.rhs_shape));

            const auto lhs_numel = numel_from_shape(broadcast_case.lhs_shape);
            const auto rhs_numel = numel_from_shape(broadcast_case.rhs_shape);

            auto lhs = utils::generate_random_array<float>(lhs_numel, 1.0F, 2.0F);
            auto rhs = utils::generate_random_array<float>(rhs_numel, 0.5F, 1.5F);
            tensor a(lhs.data(), broadcast_case.lhs_shape);
            tensor b(rhs.data(), broadcast_case.rhs_shape);

            require_op_matches_oracle(
                a.dtype(), expected_shape, [&]() { return op.comtam_op(a, b); },
                [&]() {
                    return mlx_test::binary_float32(lhs, broadcast_case.lhs_shape, rhs,
                                                    broadcast_case.rhs_shape, op.mlx_op);
                },
                ValueMode::Approximate);
        }
    }
}

/*
 * Broadcast must still work when an operand is a non-contiguous view. The binary
 * kernel indexes through physical_offset, so a transposed left operand plus a
 * broadcastable bias is a direct public proof of that path. Cover all four
 * public binary names: add/mul are primitives; sub/div are compositions.
 */
TEST_CASE("Forward compare vs MLX for broadcast binary ops on non-contiguous inputs",
          "[forward][ops][broadcast][mlx][metal]") {
    struct OpCase {
        mlx_test::BinaryOp mlx_op;
        tensor (*comtam_op)(const tensor&, const tensor&);
    };

    const OpCase op_cases[] = {
        {mlx_add, tensor::add},
        {mlx_subtract, tensor::sub},
        {mlx_multiply, tensor::mul},
        {mlx_divide, tensor::div},
    };

    // Logical: (4, 3) op (3,) -> (4, 3), with the (4, 3) operand coming from a
    // transpose of a contiguous (3, 4) base.
    const view_vector base_shape{3, 4};
    const view_vector logical_shape{4, 3};
    const view_vector bias_shape{3};

    auto base_data = utils::generate_random_array<float>(numel_from_shape(base_shape), 1.0F, 2.0F);
    auto bias_data = utils::generate_random_array<float>(numel_from_shape(bias_shape), 0.5F, 1.5F);

    tensor a_base(base_data.data(), base_shape);
    tensor a = a_base.transpose(0, 1);
    tensor b(bias_data.data(), bias_shape);

    REQUIRE(a.shape() == logical_shape);
    // Contiguous (4, 3) would be strides (3, 1); transpose of (3, 4) is (1, 4).
    REQUIRE(a.strides() == view_vector{1, 4});

    const auto a_equiv = mlx_test::transpose_float32(base_data, base_shape, {1, 0});
    const auto expected_shape = view::broadcast_shape(view(logical_shape), view(bias_shape));

    for (const auto& op : op_cases) {
        CAPTURE(op.mlx_op);
        require_op_matches_oracle(
            a.dtype(), expected_shape, [&]() { return op.comtam_op(a, b); },
            [&]() {
                return mlx_test::binary_float32(a_equiv, logical_shape, bias_data, bias_shape,
                                                op.mlx_op);
            },
            ValueMode::Approximate);
    }
}
