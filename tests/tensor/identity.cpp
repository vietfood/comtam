/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** | AI Assist : Grok 4.5 (Cursor)                                         |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/tensor/tensor.h"
#include "tests/support/forward_compare.h"
#include "tests/support/tensor_access.h"

using namespace comtam;
using comtam::tests::forward_compare::require_forward_matches;
using comtam::tests::forward_compare::ValueMode;
using comtam::tests::tensor_access;

namespace {

tensor make_sample() {
    const std::vector<float> data{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    return tensor(data.data(), {2, 3});
}

}  // namespace

TEST_CASE("copying a tensor shares logical identity", "[tensor][identity][metal]") {
    tensor a = make_sample();
    tensor b = a;

    REQUIRE(&a != &b);
    REQUIRE(a.is_same(b));
    REQUIRE(tensor_access::shares_storage(a, b));
    REQUIRE(tensor_access::shares_runtime(a, b));

    tensor c({2, 3});
    c = a;
    REQUIRE(a.is_same(c));
    REQUIRE(tensor_access::shares_storage(a, c));
}

TEST_CASE("independent construction has distinct identity and storage",
          "[tensor][identity][metal]") {
    const std::vector<float> data{1.0f, 2.0f};
    tensor a(data.data(), {2});
    tensor b(data.data(), {2});

    REQUIRE_FALSE(a.is_same(b));
    REQUIRE_FALSE(tensor_access::shares_storage(a, b));
    REQUIRE(a.to_vector<float>() == b.to_vector<float>());
}

TEST_CASE("public movement creates identity and shares storage", "[tensor][identity][metal]") {
    tensor a = make_sample();

    SECTION("reshape") {
        auto b = a.reshape({3, 2});
        REQUIRE_FALSE(a.is_same(b));
        REQUIRE(tensor_access::shares_storage(a, b));
        REQUIRE(tensor_access::shares_runtime(a, b));
        REQUIRE(b.shape() == view_vector({3, 2}));
    }

    SECTION("transpose") {
        auto b = a.transpose(0, 1);
        REQUIRE_FALSE(a.is_same(b));
        REQUIRE(tensor_access::shares_storage(a, b));
        REQUIRE(tensor_access::shares_runtime(a, b));
        REQUIRE(b.shape() == view_vector({3, 2}));
        REQUIRE(b.strides() == view_vector({1, 3}));
    }

    SECTION("permute") {
        auto b = a.permute({1, 0});
        REQUIRE_FALSE(a.is_same(b));
        REQUIRE(tensor_access::shares_storage(a, b));
        REQUIRE(tensor_access::shares_runtime(a, b));
    }

    SECTION("shrink") {
        auto b = a.shrink({{0, 2}, {1, 3}});
        REQUIRE_FALSE(a.is_same(b));
        REQUIRE(tensor_access::shares_storage(a, b));
        REQUIRE(tensor_access::shares_runtime(a, b));
        REQUIRE(b.shape() == view_vector({2, 2}));
        REQUIRE(b.offset() == 1);
    }

    SECTION("expand") {
        tensor col({3, 1});
        col.from_vector<float>({0.f, 1.f, 2.f});
        auto b = col.expand({3, 4});
        REQUIRE_FALSE(col.is_same(b));
        REQUIRE(tensor_access::shares_storage(col, b));
        REQUIRE(tensor_access::shares_runtime(col, b));
        REQUIRE(b.strides() == view_vector({1, 0}));
    }
}

TEST_CASE("handle lifetime keeps shared implementation alive", "[tensor][identity][metal]") {
    const std::vector<float> expected{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};

    tensor survivor = [&] {
        tensor original(expected.data(), {2, 3});
        tensor copy = original;
        return copy;
    }();

    REQUIRE(survivor.is_same(survivor));
    require_forward_matches<float>(survivor, {2, 3}, expected, ValueMode::Exact);
}

TEST_CASE("allocating ops create new identity and storage in operand runtime",
          "[tensor][identity][ops][metal]") {
    tensor a = make_sample();
    tensor b = make_sample();

    auto neg = tensor::neg(a);
    REQUIRE_FALSE(a.is_same(neg));
    REQUIRE_FALSE(tensor_access::shares_storage(a, neg));
    REQUIRE(tensor_access::shares_runtime(a, neg));

    auto sum = tensor::add(a, b);
    REQUIRE_FALSE(a.is_same(sum));
    REQUIRE_FALSE(b.is_same(sum));
    REQUIRE_FALSE(tensor_access::shares_storage(a, sum));
    REQUIRE(tensor_access::shares_runtime(a, sum));

    auto reduced = tensor::sum(a);
    REQUIRE_FALSE(a.is_same(reduced));
    REQUIRE_FALSE(tensor_access::shares_storage(a, reduced));
    REQUIRE(tensor_access::shares_runtime(a, reduced));

    tensor lhs({2, 3});
    tensor rhs({3, 4});
    lhs.from_vector<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    rhs.from_vector<float>(
        {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f});
    auto product = tensor::matmul(lhs, rhs);
    REQUIRE_FALSE(lhs.is_same(product));
    REQUIRE_FALSE(tensor_access::shares_storage(lhs, product));
    REQUIRE(tensor_access::shares_runtime(lhs, product));
}

TEST_CASE("tensor remains usable after its context handle is destroyed",
          "[tensor][identity][runtime][metal]") {
    const std::vector<float> data{1.f, 2.f, 3.f, 4.f};

    tensor survivor = [&] {
        core::context local;
        return tensor(data.data(), {2, 2}, DType::Float32, local);
    }();

    require_forward_matches<float>(survivor, {2, 2}, data, ValueMode::Exact);
    auto doubled = tensor::add(survivor, survivor);
    require_forward_matches<float>(doubled, {2, 2}, {2.f, 4.f, 6.f, 8.f}, ValueMode::Exact);
}

TEST_CASE("mixed-runtime binary operations reject before dispatch",
          "[tensor][identity][runtime][metal]") {
    core::context ctx_a;
    core::context ctx_b;
    REQUIRE_FALSE(ctx_a.shares_runtime_with(ctx_b));

    const std::vector<float> data{1.f, 2.f, 3.f, 4.f};
    tensor a(data.data(), {2, 2}, DType::Float32, ctx_a);
    tensor b(data.data(), {2, 2}, DType::Float32, ctx_b);

    REQUIRE_FALSE(tensor_access::shares_runtime(a, b));
    REQUIRE_THROWS_AS(tensor::add(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::sub(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::mul(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::div(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(tensor::matmul(a, b), std::runtime_error);
}

TEST_CASE("composed mean keeps scale constants in the operand runtime",
          "[tensor][identity][runtime][metal]") {
    core::context local;
    const std::vector<float> data{1.f, 2.f, 3.f, 4.f};
    tensor a(data.data(), {2, 2}, DType::Float32, local);

    auto full = tensor::mean(a);
    auto axis = tensor::mean(a, 1, false);

    REQUIRE(tensor_access::shares_runtime(a, full));
    REQUIRE(tensor_access::shares_runtime(a, axis));
    REQUIRE_FALSE(a.is_same(full));
    REQUIRE_FALSE(a.is_same(axis));
}
