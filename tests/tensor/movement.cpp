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

#include "comtam/core/context.h"
#include "comtam/tensor/tensor.h"
#include "tests/support/forward_compare.h"
#include "tests/support/mlx_oracle.h"

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::ValueMode;
namespace mlx_test = comtam::tests::mlx_oracle;

TEST_CASE("Forward compare vs MLX for movement ops", "[forward][view][mlx][metal]") {
    core::context context;
    auto& device = context.device();

    SECTION("transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        tensor tensor({2, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 2), strides=(1, 3), offset=0
        auto result = tensor.transpose(1, 0);
        require_forward_matches<float>(device, result, {3, 2},
                                       mlx_test::transpose_float32(input, {2, 3}, {1, 0}),
                                       ValueMode::Exact);
    }

    SECTION("permute") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
        tensor tensor({2, 2, 2}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(2, 2, 2), strides=(1, 4, 2), offset=0
        auto result = tensor.permute({2, 0, 1});
        require_forward_matches<float>(device, result, {2, 2, 2},
                                       mlx_test::transpose_float32(input, {2, 2, 2}, {2, 0, 1}),
                                       ValueMode::Exact);
    }

    SECTION("slice 3x3 inner 2x2") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        tensor tensor({3, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(2, 2), strides=(3, 1), offset=4
        auto result = tensor.shrink({{1, 3}, {1, 3}});
        require_forward_matches<float>(
            device, result, {2, 2}, mlx_test::slice_float32(input, {3, 3}, {1, 1}, {3, 3}, {1, 1}),
            ValueMode::Exact);
    }

    SECTION("slice 4x4 inner 2x2") {
        std::vector<float> input{0.f, 1.f, 2.f,  3.f,  4.f,  5.f,  6.f,  7.f,
                                 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
        tensor tensor({4, 4}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(2, 2), strides=(4, 1), offset=5
        auto result = tensor.shrink({{1, 3}, {1, 3}});
        require_forward_matches<float>(
            device, result, {2, 2}, mlx_test::slice_float32(input, {4, 4}, {1, 1}, {3, 3}, {1, 1}),
            ValueMode::Exact);
    }

    SECTION("expand") {
        std::vector<float> input{0.f, 1.f, 2.f};
        tensor tensor({3, 1}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 4), strides=(1, 0), offset=0
        auto result = tensor.expand({3, 4});
        require_forward_matches<float>(device, result, {3, 4},
                                       mlx_test::broadcast_to_float32(input, {3, 1}, {3, 4}),
                                       ValueMode::Exact);
    }

    SECTION("reshape") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        tensor tensor({2, 3}, device);
        tensor.from_vector<float>(input, device);

        // result view: shape=(3, 2), strides=(2, 1), offset=0
        auto result = tensor.reshape({3, 2});
        require_forward_matches<float>(device, result, {3, 2},
                                       mlx_test::reshape_float32(input, {2, 3}, {3, 2}),
                                       ValueMode::Exact);
    }

    SECTION("chained slice then transpose") {
        std::vector<float> input{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        tensor tensor({3, 3}, device);
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
