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
#include <vector>

#include "comtam/tensor/tensor.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"
#include "tests/support/tensor_access.h"

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::ValueMode;
using comtam::tests::tensor_access;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare vs MLX for movement ops", "[forward][view][mlx][metal]") {
    SECTION("transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        tensor t({2, 3});
        t.from_vector<float>(input);

        auto result = t.transpose(1, 0);
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        REQUIRE(tensor_access::shares_runtime(t, result));
        require_forward_matches<float>(result, {3, 2},
                                       mlx_test::transpose_float32(input, {2, 3}, {1, 0}),
                                       ValueMode::Exact);
    }

    SECTION("permute") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
        tensor t({2, 2, 2});
        t.from_vector<float>(input);

        auto result = t.permute({2, 0, 1});
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        REQUIRE(tensor_access::shares_runtime(t, result));
        require_forward_matches<float>(result, {2, 2, 2},
                                       mlx_test::transpose_float32(input, {2, 2, 2}, {2, 0, 1}),
                                       ValueMode::Exact);
    }

    SECTION("slice 3x3 inner 2x2") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        tensor t({3, 3});
        t.from_vector<float>(input);

        auto result = t.shrink({{1, 3}, {1, 3}});
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        REQUIRE(tensor_access::shares_runtime(t, result));
        require_forward_matches<float>(
            result, {2, 2}, mlx_test::slice_float32(input, {3, 3}, {1, 1}, {3, 3}, {1, 1}),
            ValueMode::Exact);
    }

    SECTION("slice 4x4 inner 2x2") {
        std::vector<float> input{0.f, 1.f, 2.f,  3.f,  4.f,  5.f,  6.f,  7.f,
                                 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
        tensor t({4, 4});
        t.from_vector<float>(input);

        auto result = t.shrink({{1, 3}, {1, 3}});
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        require_forward_matches<float>(
            result, {2, 2}, mlx_test::slice_float32(input, {4, 4}, {1, 1}, {3, 3}, {1, 1}),
            ValueMode::Exact);
    }

    SECTION("expand") {
        std::vector<float> input{0.f, 1.f, 2.f};
        tensor t({3, 1});
        t.from_vector<float>(input);

        auto result = t.expand({3, 4});
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        REQUIRE(tensor_access::shares_runtime(t, result));
        require_forward_matches<float>(result, {3, 4},
                                       mlx_test::broadcast_to_float32(input, {3, 1}, {3, 4}),
                                       ValueMode::Exact);
    }

    SECTION("reshape") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        tensor t({2, 3});
        t.from_vector<float>(input);

        auto result = t.reshape({3, 2});
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        REQUIRE(tensor_access::shares_runtime(t, result));
        require_forward_matches<float>(result, {3, 2},
                                       mlx_test::reshape_float32(input, {2, 3}, {3, 2}),
                                       ValueMode::Exact);
    }

    SECTION("chained slice then transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        tensor t({3, 3});
        t.from_vector<float>(input);

        auto result = t.shrink({{1, 3}, {1, 3}}).transpose(1, 0);
        REQUIRE_FALSE(t.is_same(result));
        REQUIRE(tensor_access::shares_storage(t, result));
        auto sliced = mlx_test::slice_float32(input, {3, 3}, {1, 1}, {3, 3}, {1, 1});
        require_forward_matches<float>(result, {2, 2},
                                       mlx_test::transpose_float32(sliced, {2, 2}, {1, 0}),
                                       ValueMode::Exact);
    }
}
