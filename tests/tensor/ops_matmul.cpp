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

    // Contiguous operands select the tiled 16x16 path. Include non-square and
    // tile-boundary shapes so partial tiles differ across M/N/K.
    const MatmulCase matmul_case[] = {
        {{20, 20}, {20, 20}},     // square, multi-tile with remainder
        {{20, 1}, {1, 20}},       // skinny K
        {{17, 19}, {19, 33}},     // non-square tile boundary (2x3 output tiles)
        {{5, 13}, {13, 7}},       // non-square, fits in one partial tile
        {{31, 17}, {17, 9}},      // non-square, 2x1 output tiles, multi-phase K
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

/*
 * Matmul must work when an operand is a non-contiguous view (e.g. a transpose).
 * The kernel's slow path uses physical_offset to gather strided elements, so
 * this test exercises that path against the MLX oracle. We build each
 * non-contiguous operand by transposing a contiguous base tensor (a free view
 * op, no copy); the MLX side mirrors the transpose on its own data so both
 * sides see the same logical matrix.
 */
TEST_CASE("Forward compare vs MLX for matmul on non-contiguous inputs",
          "[forward][ops][matmul][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    // C = A @ B with A: [M, K], B: [K, N], C: [M, N]. Non-square shapes are
    // intentional: they catch stride bugs that a square [20,20] would hide.
    struct NonContiguousCase {
        view_vector a_shape;  // logical shape of A
        view_vector b_shape;  // logical shape of B
        bool transpose_a;     // A = A_base.T (A_base has reversed shape)
        bool transpose_b;     // B = B_base.T
    };

    const NonContiguousCase cases[] = {
        {{12, 20}, {20, 16}, true, false},   // only A non-contiguous
        {{12, 20}, {20, 16}, false, true},   // only B non-contiguous
        {{12, 20}, {20, 16}, true, true},    // both non-contiguous
        {{7, 11}, {11, 5}, true, false},     // smaller non-square, A transposed
        {{17, 19}, {19, 33}, false, true},   // tile-boundary non-square, B transposed
        {{9, 23}, {23, 13}, true, true},     // another non-square, both transposed
    };

    for (const auto& tc : cases) {
        CAPTURE(tc.a_shape);
        CAPTURE(tc.b_shape);
        CAPTURE(tc.transpose_a);
        CAPTURE(tc.transpose_b);

        const auto M = tc.a_shape[0];
        const auto K = tc.a_shape[1];
        const auto N = tc.b_shape[1];
        REQUIRE(tc.b_shape[0] == K);

        // base shapes are the contiguous storage we actually allocate
        const view_vector a_base_shape = tc.transpose_a ? view_vector{K, M} : tc.a_shape;
        const view_vector b_base_shape = tc.transpose_b ? view_vector{N, K} : tc.b_shape;

        const auto a_base_numel = numel_from_shape(a_base_shape);
        const auto b_base_numel = numel_from_shape(b_base_shape);

        auto a_base_data = utils::generate_random_array<float>(a_base_numel, 1.0F, 2.0F);
        auto b_base_data = utils::generate_random_array<float>(b_base_numel, 0.5F, 1.5F);

        // comtam: contiguous base tensors, then transpose as a free view op
        tensor a_base(a_base_data.data(), a_base_shape, device);
        tensor b_base(b_base_data.data(), b_base_shape, device);
        tensor a = tc.transpose_a ? a_base.transpose(0, 1) : a_base;
        tensor b = tc.transpose_b ? b_base.transpose(0, 1) : b_base;

        // guard against transpose silently becoming a no-op (which would make
        // this test pass for the wrong reason): a transposed [d0, d1]
        // contiguous base has strides (1, d1), not the contiguous (d1, 1).
        // A_base is [K, M] -> strides (1, M); B_base is [N, K] -> strides (1, K).
        if (tc.transpose_a) {
            REQUIRE(a.strides() == view_vector{1, M});
        }
        if (tc.transpose_b) {
            REQUIRE(b.strides() == view_vector{1, K});
        }

        // MLX oracle: build the same logical matrices as contiguous data
        const auto a_equiv_data = tc.transpose_a
            ? mlx_test::transpose_float32(a_base_data, a_base_shape, {1, 0})
            : a_base_data;
        const auto b_equiv_data = tc.transpose_b
            ? mlx_test::transpose_float32(b_base_data, b_base_shape, {1, 0})
            : b_base_data;

        const auto expected_shape = view_vector{M, N};

        require_op_matches_oracle(
            context, a.dtype(), expected_shape, [&]() { return tensor::matmul(a, b, context); },
            [&]() {
                return mlx_test::matmul_float32(a_equiv_data, tc.a_shape, b_equiv_data, tc.b_shape);
            },
            ValueMode::Approximate);
    }
}
