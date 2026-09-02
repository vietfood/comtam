/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include "comtam/core/context.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Context creates a Metal device and kernel library", "[context][metal]") {
    comtam::core::context context;
    REQUIRE(context.device().get() != nullptr);
    REQUIRE(context.device().queue() != nullptr);
}

TEST_CASE("copied contexts share one runtime", "[context]") {
    comtam::core::context a;
    comtam::core::context b = a;

    REQUIRE(a.shares_runtime_with(b));
    REQUIRE(&a.device() == &b.device());
    REQUIRE(&a.kernels() == &b.kernels());
}

TEST_CASE("separately constructed contexts are isolated", "[context]") {
    comtam::core::context a;
    comtam::core::context b;

    REQUIRE_FALSE(a.shares_runtime_with(b));
}

TEST_CASE("default context is stable", "[context]") {
    auto& a = comtam::core::default_context();
    auto& b = comtam::core::default_context();

    REQUIRE(&a == &b);
    REQUIRE(a.shares_runtime_with(b));
}

TEST_CASE("Device should reject byte-count mismatches", "[device][metal]") {
    comtam::core::context context;
    REQUIRE(context.device().get() != nullptr);
    REQUIRE(context.device().queue() != nullptr);

    auto& device = context.device();

    // create two storage
    auto a = device.allocate(100);
    auto b = device.allocate(200);

    REQUIRE_THROWS_AS(device.copy(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(device.copy(b, a), std::runtime_error);
}
