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
#include <memory>
#include <stdexcept>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"

constexpr std::size_t kSize = 100;

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::ValueMode;

TEST_CASE("Tensor copies host data to storage and back", "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};
    tensor tensor(input.data(), shape, device);

    require_forward_matches<float>(device, tensor, shape, input, ValueMode::Exact);
}

TEST_CASE("Tensor from_vector replaces storage contents", "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};
    tensor tensor(shape, device);

    tensor.from_vector(input, device);

    require_forward_matches<float>(device, tensor, shape, input, ValueMode::Exact);
}

TEST_CASE("Tensor from_vector rejects wrong element count", "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    tensor tensor({static_cast<view_int>(kSize)}, device);
    auto wrong_size = utils::generate_random_array<float>(kSize - 1, 0.0F, 1.0F);

    REQUIRE_THROWS_AS(tensor.from_vector(wrong_size, device), std::runtime_error);
}

TEST_CASE("Two tensor headers can share one Storage safely", "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};

    std::unique_ptr<tensor> survivor = nullptr;

    {
        auto storage = std::make_shared<core::storage>(device.allocate(kSize * sizeof(float)));
        device.copy(input.data(), input.size(), *storage);

        tensor a(storage, shape, DType::Float32);
        survivor = std::make_unique<tensor>(storage, shape, DType::Float32);

        require_forward_matches<float>(device, a, shape, input, ValueMode::Exact);
        require_forward_matches<float>(device, *survivor, shape, input, ValueMode::Exact);
    }

    // survivor storage is still here
    require_forward_matches<float>(device, *survivor, shape, input, ValueMode::Exact);
}

TEST_CASE("When one Tensor write in storage, another Tensor with same storage should see it",
          "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    const view_vector shape{static_cast<view_int>(kSize)};
    auto storage = std::make_shared<core::storage>(device.allocate(kSize * sizeof(float)));

    tensor a(storage, shape, DType::Float32);
    tensor b(storage, shape, DType::Float32);

    auto replacement = utils::generate_random_array<float>(kSize, 1.0F, 2.0F);
    a.from_vector(replacement, device);
    require_forward_matches<float>(device, b, shape, replacement, ValueMode::Exact);
}

TEST_CASE("Tensor binary operations reject mismatched shapes", "[tensor][ops][api][metal]") {
    core::context context;
    auto& device = context.device();

    tensor a({2, 3}, device);
    tensor b({3, 2}, device);

    REQUIRE_THROWS_AS(tensor::add(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(a, b, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(a, b, context), std::runtime_error);
}

TEST_CASE("Tensor ops reject rank-5 inputs", "[tensor][ops][api][metal]") {
    core::context context;
    auto& device = context.device();

    tensor rank5({1, 1, 1, 1, 1}, device);
    tensor ok({2, 3}, device);

    REQUIRE_THROWS_AS(tensor::neg(rank5, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::recip(rank5, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::add(rank5, ok, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(ok, rank5, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(ok, rank5, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(rank5, context), std::runtime_error);
}

TEST_CASE("Tensor ops reject zero-extent inputs before dispatch", "[tensor][ops][api][metal]") {
    core::context context;
    auto& device = context.device();

    // allocate(0) is rejected, so build a zero-extent header over live storage.
    // Op validators must reject this before allocating an output or submitting.
    auto backing = std::make_shared<core::storage>(device.allocate(sizeof(float)));
    tensor empty(backing, view({2, 0}), DType::Float32);
    tensor ok(backing, view({2, 3}), DType::Float32);
    tensor mat_empty_a(backing, view({2, 0}), DType::Float32);
    tensor mat_empty_b(backing, view({0, 4}), DType::Float32);

    REQUIRE(empty.shape() == view_vector({2, 0}));
    REQUIRE(empty.numel() == 0);

    REQUIRE_THROWS_AS(tensor::neg(empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::recip(empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::add(empty, ok, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(ok, empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(empty, ok, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(ok, empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sum(empty, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(mat_empty_a, mat_empty_b, context), std::runtime_error);
}

TEST_CASE("tensor::matmul rejects non-2D ranks before dispatch",
          "[tensor][ops][matmul][api][metal]") {
    core::context context;
    auto& device = context.device();

    tensor rank1({3}, device);
    tensor rank2_a({2, 3}, device);
    tensor rank2_b({3, 4}, device);
    tensor rank3({1, 2, 3}, device);

    REQUIRE_THROWS_AS(tensor::matmul(rank1, rank2_b, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, rank1, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank3, rank2_b, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, rank3, context), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, tensor({4, 5}, device), context), std::runtime_error);
}

TEST_CASE("Tensor from_vector rejects non-contiguous views", "[tensor][api][view][metal]") {
    core::context context;
    auto& device = context.device();

    tensor tensor({2, 3}, device);
    tensor.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    auto transposed = tensor.transpose(1, 0);

    REQUIRE_THROWS_AS(transposed.from_vector<float>({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}, device),
                      std::runtime_error);
    require_forward_matches<float>(device, transposed, {3, 2}, {0.f, 3.f, 1.f, 4.f, 2.f, 5.f},
                                   ValueMode::Exact);
}

TEST_CASE("Tensor at and operator[] read logical elements", "[tensor][api][metal]") {
    core::context context;
    auto& device = context.device();

    tensor t({2, 3}, device);
    t.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, device);

    SECTION("contiguous linear indexing") {
        REQUIRE(t.at<float>(0) == 0.f);
        REQUIRE(t.at<float>(5) == 5.f);
        REQUIRE_THROWS_AS(t.at<float>(6), std::invalid_argument);
    }

    SECTION("non-contiguous views still use physical_offset") {
        auto transposed = t.transpose(1, 0);  // logical: [[0,3],[1,4],[2,5]]
        REQUIRE(transposed.at<float>(0) == 0.f);
        REQUIRE(transposed.at<float>(1) == 3.f);
    }

    SECTION("shrinked views honor offset") {
        auto inner = t.shrink({{0, 2}, {1, 3}});  // [[1,2],[4,5]]
        REQUIRE(inner.at<float>(0) == 1.f);
        REQUIRE(inner.at<float>(1) == 2.f);
        REQUIRE(inner.at<float>(2) == 4.f);
        REQUIRE(inner.at<float>(3) == 5.f);
    }
}
