/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** | AI assist :  Sonnet 5 (Claude)                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/
#pragma once

#include <algorithm>
#include <concepts>
#include <random>
#include <vector>

#include "comtam/macros/log.h"
#include "comtam/macros/macros.h"
#include "comtam/types.h"

namespace comtam::utils {
COMTAM_INLINE std::mt19937& rng() {
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

/**
 * generate_random_array<T>(n, lo, hi)
 *
 * Returns a vector of n elements uniformly distributed in [lo, hi].
 * Works for any arithmetic type (int, float, double, etc.).
 */
template <std::integral T>
std::vector<T> generate_random_array(size_int n, T lo, T hi) {
    COMTAM_CHECK_AND_THROW(lo <= hi, std::invalid_argument, "lo must be <= hi");
    std::uniform_int_distribution<T> dist{lo, hi};
    std::vector<T> out(n);
    std::ranges::generate(out, [&] { return dist(rng()); });
    return out;
}

template <std::floating_point T>
std::vector<T> generate_random_array(size_int n, T lo, T hi) {
    COMTAM_CHECK_AND_THROW(lo <= hi, std::invalid_argument, "lo must be <= hi");
    std::uniform_real_distribution<T> dist{lo, hi};
    std::vector<T> out(n);
    std::ranges::generate(out, [&] { return dist(rng()); });
    return out;
}
}  // namespace comtam::utils
