#include "comtam/core/view.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace comtam;

std::vector<size_t> gather_offsets(const core::View& view) {
    std::vector<size_t> offsets;
    offsets.reserve(static_cast<size_t>(view.numel()));
    for (size_t i = 0; i < static_cast<size_t>(view.numel()); ++i) {
        offsets.push_back(view.physical_offset(i));
    }
    return offsets;
}

TEST_CASE("View constructs row-major contiguous strides", "[view][strides]") {
    SECTION("scalar view is one logical element with no strides") {
        auto v = core::View({});

        REQUIRE(v.shape.empty());
        REQUIRE(v.strides.empty());
        REQUIRE(v.ref_strides.empty());
        REQUIRE(v.numel() == 1);
        REQUIRE(v.is_contiguous());
        REQUIRE(v.physical_offset(0) == 0);
        REQUIRE_THROWS_AS(v.physical_offset(1), std::invalid_argument);
    }

    SECTION("one-dimensional view has unit stride") {
        auto v = core::View({5});

        REQUIRE(v.strides == std::vector<int64_t>({1}));
        REQUIRE(v.ref_strides == std::vector<int64_t>({1}));
        REQUIRE(v.numel() == 5);
        REQUIRE(v.is_contiguous());
    }

    SECTION("three-dimensional view multiplies trailing extents") {
        auto v = core::View({2, 3, 4});

        REQUIRE(v.strides == std::vector<int64_t>({12, 4, 1}));
        REQUIRE(v.ref_strides == std::vector<int64_t>({12, 4, 1}));
        REQUIRE(v.numel() == 24);
        REQUIRE(v.is_contiguous());
    }

    SECTION("zero-size dimensions keep row-major strides but have no elements") {
        auto v = core::View({0, 3});

        REQUIRE(v.strides == std::vector<int64_t>({3, 1}));
        REQUIRE(v.ref_strides == std::vector<int64_t>({3, 1}));
        REQUIRE(v.numel() == 0);
        REQUIRE(v.is_contiguous());
        REQUIRE_THROWS_AS(v.physical_offset(0), std::invalid_argument);
    }
}

TEST_CASE("View maps logical linear indices to physical offsets", "[view][offset]") {
    SECTION("contiguous view uses linear index plus offset") {
        auto v = core::View({2, 3}, 7);

        REQUIRE(gather_offsets(v) == std::vector<size_t>({7, 8, 9, 10, 11, 12}));
    }

    SECTION("transposed strides reorder storage reads") {
        auto v = core::View({3, 2}, {1, 3});

        REQUIRE(gather_offsets(v) == std::vector<size_t>({0, 3, 1, 4, 2, 5}));
    }

    SECTION("broadcasted stride-zero dimension repeats the same storage element") {
        auto v = core::View({3, 4}, {1, 0});

        REQUIRE(gather_offsets(v) == std::vector<size_t>({0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}));
    }

    SECTION("sliced view starts from its storage offset") {
        auto v = core::View({3}, {1}, 2);

        REQUIRE(gather_offsets(v) == std::vector<size_t>({2, 3, 4}));
        REQUIRE_THROWS_AS(v.physical_offset(3), std::invalid_argument);
    }
}

TEST_CASE("View permutes axes without moving storage", "[view][permute]") {
    SECTION("general permutation reorders shape and strides") {
        auto v = core::View({2, 3, 4});

        auto y = v.permute({1, 2, 0});
        REQUIRE(y.shape == std::vector<int64_t>({3, 4, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({4, 1, 12}));
        REQUIRE_FALSE(y.is_contiguous());
        REQUIRE(y.offset == 0);
    }

    SECTION("transpose is a two-axis permutation") {
        auto v = core::View({2, 3});

        auto y = v.transpose(0, 1);
        REQUIRE(y.shape == std::vector<int64_t>({3, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({1, 3}));
        REQUIRE(gather_offsets(y) == std::vector<size_t>({0, 3, 1, 4, 2, 5}));

        auto round_trip = y.transpose(0, 1);
        REQUIRE(round_trip.shape == v.shape);
        REQUIRE(round_trip.strides == v.strides);
        REQUIRE(round_trip.is_contiguous());
    }

    SECTION("higher-rank transpose swaps only the requested axes") {
        auto v = core::View({2, 3, 4});

        auto y = v.transpose(0, 2);
        REQUIRE(y.shape == std::vector<int64_t>({4, 3, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({1, 4, 12}));

        auto round_trip = y.transpose(0, 2);
        REQUIRE(round_trip.shape == v.shape);
        REQUIRE(round_trip.strides == v.strides);
    }

    SECTION("invalid permutations fail loudly") {
        auto v = core::View({2, 3, 4});

        REQUIRE_THROWS_AS(v.permute({0, 1}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.permute({0, 1, 1}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.permute({0, 1, 3}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.transpose(-1, 1), std::invalid_argument);
        REQUIRE_THROWS_AS(v.transpose(0, 3), std::invalid_argument);
    }
}

TEST_CASE("View shrinks by changing offset and preserving strides", "[view][shrink]") {
    SECTION("contiguous 4x4 shrink extracts the inner 2x2 block") {
        auto v = core::View({4, 4});
        auto y = v.shrink({{1, 3}, {1, 3}});

        REQUIRE(y.shape == std::vector<int64_t>({2, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({4, 1}));
        REQUIRE(y.offset == 5);
        REQUIRE_FALSE(y.is_contiguous());
        REQUIRE(gather_offsets(y) == std::vector<size_t>({5, 6, 9, 10}));
    }

    SECTION("non-contiguous parent keeps its original strides") {
        auto v = core::View({3, 4}).transpose(0, 1);
        auto y = v.shrink({{1, 3}, {1, 3}});

        REQUIRE(y.shape == std::vector<int64_t>({2, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({1, 4}));
        REQUIRE(y.offset == 5);
        REQUIRE(gather_offsets(y) == std::vector<size_t>({5, 9, 6, 10}));
    }

    SECTION("invalid shrink bounds fail") {
        auto v = core::View({4, 4});

        REQUIRE_THROWS_AS(v.shrink({{1, 3}}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.shrink({{-1, 3}, {0, 2}}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.shrink({{1, 5}, {0, 2}}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.shrink({{2, 2}, {0, 2}}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.shrink({{3, 1}, {0, 2}}), std::invalid_argument);
    }
}

TEST_CASE("View expands broadcast dimensions with stride zero", "[view][expand]") {
    SECTION("trailing size-one dimension expands across columns") {
        auto v = core::View({3, 1});
        auto y = v.expand({3, 4});

        REQUIRE(y.shape == std::vector<int64_t>({3, 4}));
        REQUIRE(y.strides == std::vector<int64_t>({1, 0}));
        REQUIRE_FALSE(y.is_contiguous());
        REQUIRE(gather_offsets(y) == std::vector<size_t>({0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}));
    }

    SECTION("leading broadcast dimensions are aligned from the right") {
        auto v = core::View({3, 2});
        auto y = v.expand({5, 4, 3, 2});

        REQUIRE(y.shape == std::vector<int64_t>({5, 4, 3, 2}));
        REQUIRE(y.strides == std::vector<int64_t>({0, 0, 2, 1}));
        REQUIRE(y.physical_offset(0) == 0);
        REQUIRE(y.physical_offset(23) == 5);
        REQUIRE(y.physical_offset(47) == 5);
        REQUIRE(y.physical_offset(119) == 5);
    }

    SECTION("invalid expand shapes fail") {
        auto v = core::View({3, 2});

        REQUIRE_THROWS_AS(v.expand({3, 4}), std::invalid_argument);
        REQUIRE_THROWS_AS(v.expand({2}), std::invalid_argument);
    }
}

TEST_CASE("View reshapes contiguous views and rejects unsupported reshapes", "[view][reshape]") {
    SECTION("contiguous reshape rebuilds row-major strides and preserves offsets") {
        auto v = core::View({2, 3, 4}, 5);
        auto y = v.reshape({6, 4});

        REQUIRE(y.shape == std::vector<int64_t>({6, 4}));
        REQUIRE(y.strides == std::vector<int64_t>({4, 1}));
        REQUIRE(y.offset == 5);
        REQUIRE(y.is_contiguous());
        REQUIRE(gather_offsets(y) == gather_offsets(v));
    }

    SECTION("mismatched element counts fail") {
        auto v = core::View({2, 3, 4});

        REQUIRE_THROWS_AS(v.reshape({7, 4}), std::invalid_argument);
    }

    SECTION("non-contiguous reshape is rejected by current policy") {
        auto transposed = core::View({2, 3}).transpose(0, 1);

        REQUIRE_THROWS_AS(transposed.reshape({6}), std::invalid_argument);
    }
}

TEST_CASE("View broadcast shape", "[view][broadcast]") {
    SECTION("non-compatible shape between two views") {
        SECTION("(3, 1) and (2, 2, 4)") {
            auto v = core::View({3, 1});
            auto y = core::View({2, 2, 4});
            REQUIRE_THROWS(core::View::broadcast_shape(v, y));
        }
        SECTION("(2, 3) and (4, 3)") {
            auto v = core::View({2, 3});
            auto y = core::View({4, 3});
            REQUIRE_THROWS(core::View::broadcast_shape(v, y));
        }
    }

    SECTION("compatible view return broadcast shape") {
        SECTION("(3, 1) and (2, 1, 4) => (2, 3, 4") {
            auto v = core::View({3, 1});
            auto y = core::View({2, 1, 4});
            auto res = core::View::broadcast_shape(v, y);
            REQUIRE(res == std::vector<int64_t>({2, 3, 4}));
        }
        SECTION("(3,) and (4, 3) -> (4, 3)") {
            auto v = core::View({3});
            auto y = core::View({4, 3});
            auto res = core::View::broadcast_shape(v, y);
            REQUIRE(res == std::vector<int64_t>({4, 3}));
        }
    }
}
