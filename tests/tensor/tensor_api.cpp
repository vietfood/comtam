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
#include <stdexcept>
#include <vector>

#include "comtam/tensor/dtype.h"
#include "comtam/tensor/tensor.h"
#include "comtam/utils/rng.h"
#include "tests/support/forward_compare.h"
#include "tests/support/tensor_access.h"

constexpr std::size_t kSize = 100;

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::ValueMode;
using comtam::tests::tensor_access;

TEST_CASE("Tensor copies host data to storage and back", "[tensor][api][metal]") {
    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};
    tensor t(input.data(), shape);

    require_forward_matches<float>(t, shape, input, ValueMode::Exact);
}

TEST_CASE("Tensor from_vector replaces storage contents", "[tensor][api][metal]") {
    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};
    tensor t(shape);

    t.from_vector(input);

    require_forward_matches<float>(t, shape, input, ValueMode::Exact);
}

TEST_CASE("Tensor from_vector rejects wrong element count", "[tensor][api][metal]") {
    tensor t({static_cast<view_int>(kSize)});
    auto wrong_size = utils::generate_random_array<float>(kSize - 1, 0.0F, 1.0F);

    REQUIRE_THROWS_AS(t.from_vector(wrong_size), std::runtime_error);
}

TEST_CASE("Movement aliases share storage and observe writes", "[tensor][api][metal]") {
    auto input = utils::generate_random_array<float>(kSize, 0.0F, 1.0F);
    const view_vector shape{static_cast<view_int>(kSize)};

    tensor a(input.data(), shape);
    auto b = a.reshape(shape);

    REQUIRE_FALSE(a.is_same(b));
    REQUIRE(tensor_access::shares_storage(a, b));

    auto replacement = utils::generate_random_array<float>(kSize, 1.0F, 2.0F);
    a.from_vector(replacement);
    require_forward_matches<float>(b, shape, replacement, ValueMode::Exact);
}

TEST_CASE("Tensor binary operations reject mismatched shapes", "[tensor][ops][api][metal]") {
    tensor a({2, 3});
    tensor b({3, 2});

    REQUIRE_THROWS_AS(tensor::add(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(a, b), std::runtime_error);
}

TEST_CASE("Tensor ops reject rank-5 inputs", "[tensor][ops][api][metal]") {
    tensor rank5({1, 1, 1, 1, 1});
    tensor ok({2, 3});

    REQUIRE_THROWS_AS(tensor::neg(rank5), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::recip(rank5), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::add(rank5, ok), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(ok, rank5), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(ok, rank5), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(rank5), std::runtime_error);
}

TEST_CASE("Tensor ops reject zero-extent inputs before dispatch", "[tensor][ops][api][metal]") {
    // allocate(0) is rejected, so build a zero-extent header over live storage.
    tensor backing({2, 3});
    tensor empty = tensor_access::make_view_alias(backing, view({2, 0}));
    tensor ok = tensor_access::make_view_alias(backing, view({2, 3}));
    tensor mat_empty_a = tensor_access::make_view_alias(backing, view({2, 0}));
    tensor mat_empty_b = tensor_access::make_view_alias(backing, view({0, 4}));

    REQUIRE(empty.shape() == view_vector({2, 0}));
    REQUIRE(empty.numel() == 0);

    REQUIRE_THROWS_AS(tensor::neg(empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::recip(empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::add(empty, ok), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(ok, empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(empty, ok), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(ok, empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mean(empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sum(empty), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(mat_empty_a, mat_empty_b), std::runtime_error);
}

TEST_CASE("tensor::matmul rejects non-2D ranks before dispatch",
          "[tensor][ops][matmul][api][metal]") {
    tensor rank1({3});
    tensor rank2_a({2, 3});
    tensor rank2_b({3, 4});
    tensor rank3({1, 2, 3});

    REQUIRE_THROWS_AS(tensor::matmul(rank1, rank2_b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, rank1), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank3, rank2_b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, rank3), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(rank2_a, tensor({4, 5})), std::runtime_error);
}

TEST_CASE("Tensor from_vector rejects non-contiguous views", "[tensor][api][view][metal]") {
    tensor t({2, 3});
    t.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f});

    auto transposed = t.transpose(1, 0);

    REQUIRE_THROWS_AS(transposed.from_vector<float>({0.f, 3.f, 1.f, 4.f, 2.f, 5.f}),
                      std::runtime_error);
    require_forward_matches<float>(transposed, {3, 2}, {0.f, 3.f, 1.f, 4.f, 2.f, 5.f},
                                   ValueMode::Exact);
}

TEST_CASE("Tensor at reads logical elements", "[tensor][api][metal]") {
    tensor t({2, 3});
    t.from_vector<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f});

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
