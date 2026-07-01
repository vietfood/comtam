#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace comtam::core {
using ViewInt = int64_t;
using ViewVector = std::vector<ViewInt>;
using PairViewVector = std::vector<std::pair<ViewInt, ViewInt>>;

struct View {
    ViewVector shape;
    ViewVector strides;
    ViewInt offset = 0;

    // for contiguous check
    ViewVector ref_strides;

    // it should compare from shape -> strides -> offset
    bool operator==(const View& other) const = default;

    // --- Constructor ---
    View(const ViewVector& shape, ViewInt offset = 0);
    View(const ViewVector& shape, const ViewVector& strides, ViewInt offset = 0);

    // --- Getter ---
    bool is_contiguous() const { return strides == ref_strides; }

    size_t dim() const { return shape.size(); }

    ViewInt numel() const;

    // returns offset + sum(coord[d] * strides[d]) for a flat linear index
    size_t physical_offset(size_t linear_index) const;

    // --- Methods ---

    /*
     * - If shape is (a, b, c) and new_axis is (0, 2, 1),
     * then the new shape is (a, c, b).
     * - Reference: https://pytorch.org/docs/stable/generated/torch.permute.html
     */
    View permute(const ViewVector& new_axis) const;

    View transpose(ViewInt a, ViewInt b) const;

    /**
     * Shrink each axis to [start, end)
     *
     * - For example: `auto y = x.shrink({{1, 3}, {1, 3}});`
     * - Then: axis 0 keeps rows [1, 3), and axis 1 keeps cols [1, 3).
     */
    View shrink(const PairViewVector& limits) const;

    View expand(const ViewVector& new_shape) const;

    View reshape(const ViewVector& new_shape) const;

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
    static ViewVector broadcast_shape(const View& lhs, const View& rhs);
};
}  // namespace comtam::core
