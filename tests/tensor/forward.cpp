#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::require_op_matches_oracle;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare helper", "[forward]") {
    core::Context context;
    auto& device = context.device();

    SECTION("exact host round-trip") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        Tensor tensor(input.data(), {2, 3}, device);
        require_forward_matches<float>(device, tensor, {2, 3}, input, ValueMode::Exact);
    }

    SECTION("approximate arithmetic") {
        std::vector<float> lhs{1.f, 2.f, 3.f, 4.f};
        std::vector<float> rhs{10.f, 20.f, 30.f, 40.f};
        Tensor a(lhs.data(), {2, 2}, device);
        Tensor b(rhs.data(), {2, 2}, device);

        require_op_matches_oracle(
            context, a.dtype(), {2, 2}, [&]() { return Tensor::add(a, b, context); },
            [&]() { return std::vector<float>{11.f, 22.f, 33.f, 44.f}; }, ValueMode::Approximate);
    }
}

TEST_CASE("Forward compare vs MLX for elementwise ops", "[forward][mlx][metal]") {
    core::Context context;
    auto& device = context.device();

    struct OpCase {
        mlx_test::BinaryOp mlx_op;
        Tensor (*comtam_op)(const Tensor&, const Tensor&, core::Context&);
    };

    const OpCase cases[] = {
        {mlx_add, Tensor::add},
        {mlx_subtract, Tensor::sub},
        {mlx_multiply, Tensor::mul},
        {mlx_divide, Tensor::div},
    };

    struct ShapeCase {
        std::vector<core::ViewInt> shape;
        size_t numel;
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
            Tensor a(lhs.data(), shape, device);
            Tensor b(rhs.data(), shape, device);

            require_op_matches_oracle(
                context, a.dtype(), shape, [&]() { return op.comtam_op(a, b, context); },
                [&]() { return mlx_test::binary_float32(lhs, rhs, shape, op.mlx_op); },
                ValueMode::Approximate);
        }
    }
}

TEST_CASE("Forward compare vs MLX for movement ops", "[forward][mlx][metal]") {
    core::Context context;
    auto& device = context.device();

    SECTION("transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        Tensor tensor({2, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 2), strides=(1, 3), offset=0
        auto result = tensor.transpose(1, 0);
        require_forward_matches<float>(device, result, {3, 2},
                                       mlx_test::transpose_float32(input, {2, 3}, {1, 0}),
                                       ValueMode::Exact);
    }

    SECTION("permute") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
        Tensor tensor({2, 2, 2}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(2, 2, 2), strides=(1, 4, 2), offset=0
        auto result = tensor.permute({2, 0, 1});
        require_forward_matches<float>(device, result, {2, 2, 2},
                                       mlx_test::transpose_float32(input, {2, 2, 2}, {2, 0, 1}),
                                       ValueMode::Exact);
    }

    SECTION("slice") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        Tensor tensor({3, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(2, 2), strides=(3, 1), offset=4
        auto result = tensor.shrink({{1, 3}, {1, 3}});
        require_forward_matches<float>(
            device, result, {2, 2}, mlx_test::slice_float32(input, {3, 3}, {1, 1}, {3, 3}, {1, 1}),
            ValueMode::Exact);
    }

    SECTION("expand") {
        std::vector<float> input{0.f, 1.f, 2.f};
        Tensor tensor({3, 1}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 4), strides=(1, 0), offset=0
        auto result = tensor.expand({3, 4});
        require_forward_matches<float>(device, result, {3, 4},
                                       mlx_test::broadcast_to_float32(input, {3, 1}, {3, 4}),
                                       ValueMode::Exact);
    }

    SECTION("reshape") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        Tensor tensor({2, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 2), strides=(2, 1), offset=0
        auto result = tensor.reshape({3, 2});
        require_forward_matches<float>(device, result, {3, 2},
                                       mlx_test::reshape_float32(input, {2, 3}, {3, 2}),
                                       ValueMode::Exact);
    }

    SECTION("chained slice then transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        Tensor tensor({3, 3}, device);
        tensor.from_vector<float>(input, device);

        // after shrink: shape=(2, 2), strides=(3, 1), offset=4
        // after transpose: shape=(2, 2), strides=(1, 3), offset=4
        auto result = tensor.shrink({{1, 3}, {1, 3}}).transpose(1, 0);
        auto sliced = mlx_test::slice_float32(input, {3, 3}, {1, 1}, {3, 3}, {1, 1});
        require_forward_matches<float>(device, result, {2, 2},
                                       mlx_test::transpose_float32(sliced, {2, 2}, {1, 0}),
                                       ValueMode::Exact);
    }
}
