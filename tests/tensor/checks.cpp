/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "comtam/tensor/checks.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/view.h"

using namespace comtam;

namespace {

constexpr DType kBadDtype = static_cast<DType>(999);

}  // namespace

TEST_CASE("check_supported_dtype accepts float32 and rejects unknown", "[checks]") {
    REQUIRE_NOTHROW(checks::check_supported_dtype(DType::Float32));
    REQUIRE_THROWS_AS(checks::check_supported_dtype(kBadDtype), std::runtime_error);
}

TEST_CASE("check_rank_le enforces the GPU descriptor limit", "[checks]") {
    REQUIRE_NOTHROW(checks::check_rank_le(view({})));
    REQUIRE_NOTHROW(checks::check_rank_le(view({2, 3, 4, 5})));
    REQUIRE_THROWS_AS(checks::check_rank_le(view({1, 1, 1, 1, 1})), std::runtime_error);
}

TEST_CASE("check_positive_extents rejects zero and negative dims", "[checks]") {
    REQUIRE_NOTHROW(checks::check_positive_extents(view({})));
    REQUIRE_NOTHROW(checks::check_positive_extents(view({2, 3})));
    REQUIRE_THROWS_AS(checks::check_positive_extents(view({2, 0})), std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_positive_extents(view({-1})), std::runtime_error);
}

TEST_CASE("check_unary validates dtype rank and extents", "[checks]") {
    REQUIRE_NOTHROW(checks::check_unary(view({2, 3}), DType::Float32));
    REQUIRE_THROWS_AS(checks::check_unary(view({1, 1, 1, 1, 1}), DType::Float32),
                      std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_unary(view({2, 0}), DType::Float32), std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_unary(view({2, 3}), kBadDtype), std::runtime_error);
}

TEST_CASE("check_binary returns broadcast shape and rejects bad metadata", "[checks]") {
    const view a({4, 1});
    const view b({3});
    REQUIRE(checks::check_binary(a, DType::Float32, b, DType::Float32) == view_vector({4, 3}));

    REQUIRE_THROWS_AS(checks::check_binary(a, DType::Float32, b, kBadDtype), std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_binary(view({2, 3}), DType::Float32, view({3, 2}), DType::Float32),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_binary(view({1, 1, 1, 1, 1}), DType::Float32, view({1}), DType::Float32),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_binary(view({2, 0}), DType::Float32, view({2, 1}), DType::Float32),
        std::runtime_error);
}

TEST_CASE("check_binary_scalar validates the tensor operand", "[checks]") {
    REQUIRE_NOTHROW(checks::check_binary_scalar(view({2, 3}), DType::Float32));
    REQUIRE_THROWS_AS(checks::check_binary_scalar(view({0, 3}), DType::Float32),
                      std::runtime_error);
}

TEST_CASE("check_reduce_full and check_reduce_axis enforce reduce contracts", "[checks]") {
    const view contiguous({2, 3});
    REQUIRE(checks::check_reduce_full(contiguous, DType::Float32).empty());
    REQUIRE(checks::check_reduce_axis(contiguous, DType::Float32, 1, false) == view_vector({2}));
    REQUIRE(checks::check_reduce_axis(contiguous, DType::Float32, 1, true) == view_vector({2, 1}));

    const view non_contiguous({2, 3}, {1, 2}, 0);
    REQUIRE_FALSE(non_contiguous.is_contiguous());
    REQUIRE_THROWS_AS(checks::check_reduce_full(non_contiguous, DType::Float32),
                      std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_reduce_axis(non_contiguous, DType::Float32, 0, false),
                      std::runtime_error);

    REQUIRE_THROWS_AS(checks::check_reduce_axis(view({}), DType::Float32, 0, false),
                      std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_reduce_axis(contiguous, DType::Float32, -1, false),
                      std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_reduce_axis(contiguous, DType::Float32, 2, false),
                      std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_reduce_full(view({2, 0}), DType::Float32), std::runtime_error);
    REQUIRE_THROWS_AS(checks::check_reduce_full(view({1, 1, 1, 1, 1}), DType::Float32),
                      std::runtime_error);
}

TEST_CASE("check_matmul validates 2D positive shapes and inner dims", "[checks]") {
    REQUIRE(checks::check_matmul(view({2, 3}), DType::Float32, view({3, 4}), DType::Float32) ==
            view_vector({2, 4}));

    REQUIRE_THROWS_AS(
        checks::check_matmul(view({2, 3}), DType::Float32, view({3, 4}), kBadDtype),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_matmul(view({3}), DType::Float32, view({3, 4}), DType::Float32),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_matmul(view({2, 3}), DType::Float32, view({2, 4}), DType::Float32),
        std::runtime_error);
    REQUIRE_THROWS_AS(
        checks::check_matmul(view({2, 0}), DType::Float32, view({0, 4}), DType::Float32),
        std::runtime_error);
}
