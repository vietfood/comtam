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
#include <cmath>
#include <limits>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "mlx/c/ops.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::FloatClass;
using comtam::tests::forward_compare::require_float_class;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::require_values_close_abs_rel;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

namespace {

constexpr float kPosZero = 0.0F;
constexpr float kNegZero = -0.0F;
constexpr float kPosInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

float run_unary(core::context& context, float input,
                tensor (*op)(const tensor&, core::context&)) {
    tensor a(&input, {}, context.device());
    auto out = op(a, context);
    return out.to_vector<float>(context.device()).at(0);
}

float run_binary(core::context& context, float lhs, float rhs,
                 tensor (*op)(const tensor&, const tensor&, core::context&)) {
    tensor a(&lhs, {}, context.device());
    tensor b(&rhs, {}, context.device());
    auto out = op(a, b, context);
    return out.to_vector<float>(context.device()).at(0);
}

}  // namespace

TEST_CASE("neg preserves IEEE signed zeros, infinities, and NaN",
          "[forward][ops][numerics][metal]") {
    core::context context;

    require_float_class(run_unary(context, kPosZero, tensor::neg), FloatClass::NegZero);
    require_float_class(run_unary(context, kNegZero, tensor::neg), FloatClass::PosZero);
    require_float_class(run_unary(context, kPosInf, tensor::neg), FloatClass::NegInf);
    require_float_class(run_unary(context, kNegInf, tensor::neg), FloatClass::PosInf);
    require_float_class(run_unary(context, kNaN, tensor::neg), FloatClass::NaN);
    require_float_class(run_unary(context, 2.5F, tensor::neg), FloatClass::Finite);
    REQUIRE(run_unary(context, 2.5F, tensor::neg) == -2.5F);
}

TEST_CASE("recip follows IEEE float32 edges", "[forward][ops][numerics][metal]") {
    core::context context;

    require_float_class(run_unary(context, kPosZero, tensor::recip), FloatClass::PosInf);
    require_float_class(run_unary(context, kNegZero, tensor::recip), FloatClass::NegInf);
    require_float_class(run_unary(context, kPosInf, tensor::recip), FloatClass::PosZero);
    require_float_class(run_unary(context, kNegInf, tensor::recip), FloatClass::NegZero);
    require_float_class(run_unary(context, kNaN, tensor::recip), FloatClass::NaN);

    // Finite accuracy away from zero against MLX.
    const view_vector shape{3, 4};
    auto data = utils::generate_random_array<float>(12, 0.5F, 2.0F);
    tensor a(data.data(), shape, context.device());
    require_op_matches_oracle(
        context, a.dtype(), shape, [&]() { return tensor::recip(a, context); },
        [&]() { return mlx_test::unary_float32(data, shape, mlx_reciprocal); },
        ValueMode::Approximate);
}

TEST_CASE("tensor div follows IEEE float32 edges via mul(recip)",
          "[forward][ops][numerics][metal]") {
    core::context context;

    struct Case {
        float lhs;
        float rhs;
        FloatClass expected;
    };

    const Case cases[] = {
        {1.0F, kPosZero, FloatClass::PosInf},
        {-1.0F, kPosZero, FloatClass::NegInf},
        {1.0F, kNegZero, FloatClass::NegInf},
        {-1.0F, kNegZero, FloatClass::PosInf},
        {kPosZero, kPosZero, FloatClass::NaN},
        {kNegZero, kNegZero, FloatClass::NaN},
        {kPosZero, 2.0F, FloatClass::PosZero},
        {kNegZero, 2.0F, FloatClass::NegZero},
        {kPosZero, kPosInf, FloatClass::PosZero},
        {2.0F, kPosInf, FloatClass::PosZero},
        {-2.0F, kPosInf, FloatClass::NegZero},
        {kPosInf, 2.0F, FloatClass::PosInf},
        {kNegInf, 2.0F, FloatClass::NegInf},
        {kPosInf, kPosInf, FloatClass::NaN},
        {kNaN, 2.0F, FloatClass::NaN},
        {2.0F, kNaN, FloatClass::NaN},
    };

    for (const auto& c : cases) {
        CAPTURE(c.lhs);
        CAPTURE(c.rhs);
        require_float_class(run_binary(context, c.lhs, c.rhs, tensor::div), c.expected);
    }
}

TEST_CASE("finite recip and composed div stay within abs+rel tolerance vs MLX",
          "[forward][ops][numerics][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    constexpr double kAbsEps = 1e-5;
    constexpr double kRelEps = 1e-5;

    const view_vector shape{4, 5};
    auto lhs = utils::generate_random_array<float>(20, 0.5F, 2.0F);
    auto rhs = utils::generate_random_array<float>(20, 0.5F, 2.0F);
    tensor a(lhs.data(), shape, device);
    tensor b(rhs.data(), shape, device);

    {
        auto actual = tensor::recip(a, context).to_vector<float>(device);
        auto expected = mlx_test::unary_float32(lhs, shape, mlx_reciprocal);
        require_values_close_abs_rel(expected, actual, kAbsEps, kRelEps, shape);
    }

    {
        auto actual = tensor::div(a, b, context).to_vector<float>(device);
        auto expected = mlx_test::binary_float32(lhs, shape, rhs, shape, mlx_divide);
        // Public div is mul(a, recip(b)); allow the same single abs+rel budget vs MLX divide.
        require_values_close_abs_rel(expected, actual, kAbsEps, kRelEps, shape);
    }
}
