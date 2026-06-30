#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "comtam/core/context.h"
#include "comtam/core/dtype.h"
#include "comtam/core/storage.h"
#include "comtam/tensor.h"
#include "comtam/utils/cpu_op.h"
#include "comtam/utils/rng.h"

#include <memory>
#include <stdexcept>
#include <vector>

constexpr std::size_t kSize = 100;

using namespace comtam;

void require_all_close(const std::vector<float> &expected, const std::vector<float> &actual,
                       float epsilon = 1e-5F) {
    REQUIRE(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE_THAT(actual[i], Catch::Matchers::WithinAbs(expected[i], epsilon));
    }
}

TEST_CASE("Tensor copies host data to storage and back", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    Tensor tensor(input.data(), {static_cast<core::ViewInt>(kSize)}, device);

    require_all_close(input, tensor.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector replaces storage contents", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    Tensor tensor({static_cast<core::ViewInt>(kSize)}, device);

    tensor.from_vector(input, device);

    require_all_close(input, tensor.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector rejects wrong element count", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

    Tensor tensor({static_cast<core::ViewInt>(kSize)}, device);
    auto wrong_size = utils::generate_random_array<float>(kSize - 1, 0.0F, 1.0F);

    REQUIRE_THROWS_AS(tensor.from_vector(wrong_size, device), std::runtime_error);
}

TEST_CASE("Two tensor headers can share one Storage safely", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

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

TEST_CASE("When one Tensor write in storage, another Tensor with same storage should see it", "[tensor][metal]") {
    core::Context context;
    auto &device = context.device();

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
TEST_CASE("Tensor add matches CPU oracle for same-shape vectors", "[tensor][ops][metal]") {
    core::Context context;
    auto &device = context.device();
    auto &kernels = context.kernels();

    auto lhs = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    auto rhs = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);

    Tensor a(lhs.data(), {static_cast<core::ViewInt>(kSize)}, device);
    Tensor b(rhs.data(), {static_cast<core::ViewInt>(kSize)}, device);

    std::vector<float> expected(kSize);
    utils::add(lhs.data(), rhs.data(), expected.data(), expected.size());

    Tensor result = Tensor::add(a, b, device, kernels);

    require_all_close(expected, result.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through non-contiguous views", "[tensor][view][metal]") {
    core::Context context;
    auto &device = context.device();

    // [[0, 1, 2],
    // [3, 4, 5]]
    Tensor tensor({2, 3}, device);
    tensor.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    // Transposed logical values:
    // [[0, 3],
    // [1, 4],
    // [2, 5]]
    auto transposed = tensor.transpose(1, 0);

    require_all_close({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, transposed.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through shrink views", "[tensor][view][metal]") {
    core::Context context;
    auto &device = context.device();

    Tensor tensor({4, 4}, device);
    tensor.from_vector<float>(
        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
         13.f, 14.f, 15.f},
        device);

    auto inner = tensor.shrink({{1, 3}, {1, 3}});

    require_all_close({5.f, 6.f, 9.f, 10.f}, inner.to_vector<float>(device));
}

TEST_CASE("Tensor to_vector gathers through expand views", "[tensor][view][metal]") {
    core::Context context;
    auto &device = context.device();

    Tensor tensor({3, 1}, device);
    tensor.from_vector<float>({0.f, 1.f, 2.f}, device);

    auto expanded = tensor.expand({3, 4});

    require_all_close({0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f},
                      expanded.to_vector<float>(device));
}

TEST_CASE("Tensor from_vector rejects non-contiguous views", "[tensor][view][metal]") {
    core::Context context;
    auto &device = context.device();

    Tensor tensor({2, 3}, device);
    tensor.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    auto transposed = tensor.transpose(1, 0);

    REQUIRE_THROWS_AS(
        transposed.from_vector<float>({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, device),
        std::runtime_error);
    require_all_close({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, transposed.to_vector<float>(device));
}
