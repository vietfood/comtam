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
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::numel_from_shape;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare vs MLX for matmul", "[forward][ops][matmul][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    struct MatmulCase {
        view_vector lhs_shape;
        view_vector rhs_shape;
    };

    const MatmulCase matmul_case[] = {
        {{20, 20}, {20, 20}},
        {{20, 1}, {1, 20}},
    };

    for (const auto& shape : matmul_case) {
        CAPTURE(shape.lhs_shape);
        CAPTURE(shape.rhs_shape);

        const auto expected_shape = view_vector({shape.lhs_shape[0], shape.rhs_shape[1]});

        const auto lhs_numel = numel_from_shape(shape.lhs_shape);
        const auto rhs_numel = numel_from_shape(shape.rhs_shape);

        auto lhs = utils::generate_random_array<float>(lhs_numel, 1.0F, 2.0F);
        auto rhs = utils::generate_random_array<float>(rhs_numel, 0.5F, 1.5F);

        tensor a(lhs.data(), shape.lhs_shape, device);
        tensor b(rhs.data(), shape.rhs_shape, device);

        require_op_matches_oracle(
            context, a.dtype(), expected_shape, [&]() { return tensor::matmul(a, b, context); },
            [&]() { return mlx_test::matmul_float32(lhs, shape.lhs_shape, rhs, shape.rhs_shape); },
            ValueMode::Approximate);
    }
}
