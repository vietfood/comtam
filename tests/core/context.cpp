#include "comtam/core/context.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Context creates a Metal device and kernel library", "[context][metal]") {
    comtam::core::Context context;
    REQUIRE(context.device().get() != nullptr);
    REQUIRE(context.device().queue() != nullptr);
}

TEST_CASE("Device should reject byte-count mismatches", "[device][metal]") {
    comtam::core::Context context;
    REQUIRE(context.device().get() != nullptr);
    REQUIRE(context.device().queue() != nullptr);

    auto& device = context.device();

    // create two storage
    auto a = device.allocate(100);
    auto b = device.allocate(200);

    REQUIRE_THROWS_AS(device.copy(a, b), std::runtime_error);
    REQUIRE_THROWS_AS(device.copy(b, a), std::runtime_error);
}
