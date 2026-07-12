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

#include "comtam/core/context.h"
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
    core::context context;
    auto& device = context.device();

    struct OpCase {
        mlx_test::BinaryOp mlx_op;
        tensor (*comtam_op)(const tensor&, const tensor&, core::context&);
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
            tensor a(lhs.data(), broadcast_case.lhs_shape, device);
            tensor b(rhs.data(), broadcast_case.rhs_shape, device);

            require_op_matches_oracle(
                context, a.dtype(), expected_shape, [&]() { return op.comtam_op(a, b, context); },
                [&]() {
                    return mlx_test::binary_broadcast_float32(lhs, broadcast_case.lhs_shape, rhs,
                                                              broadcast_case.rhs_shape, op.mlx_op);
                },
                ValueMode::Approximate);
        }
    }
}
