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

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "comtam/types.h"

namespace comtam {
using view_vector = std::vector<view_int>;
using pair_view_vector = std::vector<std::pair<view_int, view_int>>;

struct view {
    view_vector shape;
    view_vector strides;
    view_int offset = 0;

    // for contiguous check
    view_vector ref_strides;

    // it should compare from shape -> strides -> offset
    bool operator==(const view& other) const = default;

    // --- Constructor ---
    view(const view_vector& shape, view_int offset = 0);
    view(const view_vector& shape, const view_vector& strides, view_int offset = 0);

    // --- Getter ---
    bool is_contiguous() const { return strides == ref_strides; }

    view_int dim() const { return static_cast<view_int>(shape.size()); }

    view_int numel() const;

    /*
     * Returns offset + sum(coord[d] * strides[d]) for a flat linear index.
     * Logical index in (view_int), physical storage offset out (size_int).
     */
    size_int physical_offset(view_int linear_index) const;

    // --- Methods ---

    /*
     * - If shape is (a, b, c) and new_axis is (0, 2, 1),
     * then the new shape is (a, c, b).
     * - Reference: https://pytorch.org/docs/stable/generated/torch.permute.html
     */
    view permute(const view_vector& new_axis) const;

    view transpose(view_int a, view_int b) const;

    /**
     * Shrink each axis to [start, end)
     *
     * - For example: `auto y = x.shrink({{1, 3}, {1, 3}});`
     * - Then: axis 0 keeps rows [1, 3), and axis 1 keeps cols [1, 3).
     */
    view shrink(const pair_view_vector& limits) const;

    view expand(const view_vector& new_shape) const;

    view reshape(const view_vector& new_shape) const;

    /*
     * Return broadcast shape between two shapes, throw error if they aren't compatible
     *
     * Based on numpy rule, we starts with the trailing (i.e. rightmost) dimension
     * and works its way left. Two dimensions are compatible when:
     * - they are equal, or
     * - one of them is 1.
     *
     * Reference: https://numpy.org/doc/stable/user/basics.broadcasting.html
     */
    static view_vector broadcast_shape(const view& lhs, const view& rhs);
};
}  // namespace comtam
