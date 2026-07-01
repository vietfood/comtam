#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/dtype.h"
#include "comtam/core/storage.h"
#include "comtam/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/mlx_oracle.h"

constexpr std::size_t kSize = 100;

using namespace comtam;

void require_all_close(const std::vector<float>& expected, const std::vector<float>& actual,
                       float epsilon = 1e-5F) {
    REQUIRE(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], epsilon));
    }
}

std::size_t numel_from_shape(const std::vector<core::ViewInt>& shape) {
    return static_cast<std::size_t>(
        std::reduce(shape.begin(), shape.end(), core::ViewInt{1}, std::multiplies<>()));
}

void require_binary_ops_match_mlx_oracles(const std::vector<core::ViewInt>& shape,
                                          core::Context& context) {
    auto& device = context.device();
    auto numel = numel_from_shape(shape);
    auto lhs = utils::generate_random_array<float>(numel, 1.0F, 2.0F);
    auto rhs = utils::generate_random_array<float>(numel, 0.5F, 1.5F);

    Tensor a(lhs.data(), shape, device);
    Tensor b(rhs.data(), shape, device);

    auto expected = tests::mlx_oracle::binary_float32(lhs, rhs, shape, mlx_add);
    require_all_close(expected, Tensor::add(a, b, context).to_vector<float>(device));

    expected = tests::mlx_oracle::binary_float32(lhs, rhs, shape, mlx_subtract);
    require_all_close(expected, Tensor::sub(a, b, context).to_vector<float>(device));

    expected = tests::mlx_oracle::binary_float32(lhs, rhs, shape, mlx_multiply);
    require_all_close(expected, Tensor::mul(a, b, context).to_vector<float>(device));

    expected = tests::mlx_oracle::binary_float32(lhs, rhs, shape, mlx_divide);
    require_all_close(expected, Tensor::div(a, b, context).to_vector<float>(device));
}

TEST_CASE("Tensor copies host data to storage and back", "[tensor][metal]") {
    core::Context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    Tensor tensor(input.data(), {static_cast<core::ViewInt>(kSize)}, device);

    require_all_close(input, tensor.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector replaces storage contents", "[tensor][metal]") {
    core::Context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    Tensor tensor({static_cast<core::ViewInt>(kSize)}, device);

    tensor.from_vector(input, device);

    require_all_close(input, tensor.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector rejects wrong element count", "[tensor][metal]") {
    core::Context context;
    auto& device = context.device();

    Tensor tensor({static_cast<core::ViewInt>(kSize)}, device);
    auto wrong_size = utils::generate_random_array<float>(kSize - 1, 0.0F, 1.0F);

    REQUIRE_THROWS_AS(tensor.from_vector(wrong_size, device), std::runtime_error);
}

TEST_CASE("Two tensor headers can share one Storage safely", "[tensor][metal]") {
    core::Context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    auto shape = std::vector<core::ViewInt>{static_cast<core::ViewInt>(kSize)};

    std::unique_ptr<Tensor> survivor = nullptr;

    {
        auto storage = std::make_shared<core::Storage>(device.allocate(kSize * sizeof(float)));
        device.copy(input.data(), input.size(), *storage);

        Tensor a(storage, shape, core::DType::Float32);
        survivor = std::make_unique<Tensor>(storage, shape, core::DType::Float32);

        require_all_close(input, a.to_vector<float>(device));
        require_all_close(input, survivor->to_vector<float>(device));
    }

    // survivor storage is still here
    require_all_close(input, survivor->to_vector<float>(device));
}

TEST_CASE("When one Tensor write in storage, another Tensor with same storage should see it",
          "[tensor][metal]") {
    core::Context context;
    auto& device = context.device();

    auto shape = std::vector<core::ViewInt>{static_cast<core::ViewInt>(kSize)};
    auto storage = std::make_shared<core::Storage>(device.allocate(kSize * sizeof(float)));

    Tensor a(storage, shape, core::DType::Float32);
    Tensor b(storage, shape, core::DType::Float32);

    auto replacement = utils::generate_random_array<float>(kSize, 1.0F, 2.0F);
    a.from_vector(replacement, device);
    require_all_close(replacement, b.to_vector<float>(device));
}

/*
 * Operation
 */
TEST_CASE("Tensor binary operations match external oracles for asymmetric shapes",
          "[tensor][ops][metal]") {
    core::Context context;

    SECTION("2x3") {
        require_binary_ops_match_mlx_oracles({2, 3}, context);
    }

    SECTION("3x4") {
        require_binary_ops_match_mlx_oracles({3, 4}, context);
    }

    SECTION("2x3x4") {
        require_binary_ops_match_mlx_oracles({2, 3, 4}, context);
    }
}

TEST_CASE("Tensor binary operations reject mismatched shapes", "[tensor][ops][metal]") {
    core::Context context;
    auto& device = context.device();

    Tensor a({2, 3}, device);
    Tensor b({3, 2}, device);

    REQUIRE_THROWS_AS(Tensor::add(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::sub(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::mul(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::div(a, b, context), std::runtime_error);
}

TEST_CASE("Tensor binary operations reject non-contiguous inputs", "[tensor][ops][metal]") {
    core::Context context;
    auto& device = context.device();

    Tensor base({2, 3}, device);
    base.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    auto non_contiguous = base.transpose(1, 0);
    Tensor contiguous({3, 2}, device);
    contiguous.from_vector<float>({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, device);

    REQUIRE_THROWS_AS(Tensor::add(non_contiguous, contiguous, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::sub(contiguous, non_contiguous, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::mul(non_contiguous, contiguous, context), std::runtime_error);
    REQUIRE_THROWS_AS(Tensor::div(contiguous, non_contiguous, context), std::runtime_error);
}

TEST_CASE("Tensor to_vector gathers through non-contiguous views", "[tensor][view][metal]") {
    core::Context context;
    auto& device = context.device();

    std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};

    // [[0, 1, 2],
    // [3, 4, 5]]
    Tensor tensor({2, 3}, device);
    tensor.from_vector<float>(input, device);

    // Transposed logical values:
    // [[0, 3],
    // [1, 4],
    // [2, 5]]
    auto transposed = tensor.transpose(1, 0);

    auto expected = tests::mlx_oracle::transpose_float32(input, {2, 3}, {1, 0});

    require_all_close(expected, transposed.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through shrink views", "[tensor][view][metal]") {
    core::Context context;
    auto& device = context.device();

    std::vector<float> input{0.f, 1.f, 2.f,  3.f,  4.f,  5.f,  6.f,  7.f,
                             8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};

    Tensor tensor({4, 4}, device);
    tensor.from_vector<float>(input, device);

    auto inner = tensor.shrink({{1, 3}, {1, 3}});

    auto expected = tests::mlx_oracle::slice_float32(input, {4, 4}, {1, 1}, {3, 3}, {1, 1});

    require_all_close(expected, inner.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through expand views", "[tensor][view][metal]") {
    core::Context context;
    auto& device = context.device();

    std::vector<float> input{0.f, 1.f, 2.f};

    Tensor tensor({3, 1}, device);
    tensor.from_vector<float>(input, device);

    auto expanded = tensor.expand({3, 4});

    auto expected = tests::mlx_oracle::broadcast_to_float32(input, {3, 1}, {3, 4});

    require_all_close(expected, expanded.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through reshape views", "[tensor][view][metal]") {
    core::Context context;
    auto& device = context.device();

    std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};

    Tensor tensor({2, 3}, device);
    tensor.from_vector<float>(input, device);

    auto reshaped = tensor.reshape({3, 2});

    auto expected = tests::mlx_oracle::reshape_float32(input, {2, 3}, {3, 2});

    require_all_close(expected, reshaped.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector rejects non-contiguous views", "[tensor][view][metal]") {
    core::Context context;
    auto& device = context.device();

    Tensor tensor({2, 3}, device);
    tensor.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    auto transposed = tensor.transpose(1, 0);

    REQUIRE_THROWS_AS(transposed.from_vector<float>({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, device),
                      std::runtime_error);
    require_all_close({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, transposed.to_vector<float>(device));
}
