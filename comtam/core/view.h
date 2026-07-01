#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <tuple>
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
    View(const ViewVector &shape, ViewInt offset = 0)
        : shape(shape), strides(shape.size(), 1), offset(offset), ref_strides(shape.size(), 1) {
        for (ViewInt i = static_cast<ViewInt>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
            ref_strides[i] = ref_strides[i + 1] * shape[i + 1];
        }
    }

    View(const ViewVector &shape, const ViewVector &strides, ViewInt offset = 0)
        : shape(shape), strides(strides), offset(offset), ref_strides(shape.size(), 1) {
        if (shape.size() != strides.size()) {
            throw std::runtime_error("View: shape and strides must have the same rank");
        }

        for (ViewInt i = static_cast<ViewInt>(shape.size()) - 2; i >= 0; --i) {
            ref_strides[i] = ref_strides[i + 1] * shape[i + 1];
        }
    }

    // --- Getter ---
    bool is_contiguous() const {
        return strides == ref_strides;
    }

    size_t dim() const {
        return shape.size();
    }

    ViewInt numel() const {
        return std::reduce(shape.begin(), shape.end(), ViewInt{1}, std::multiplies<>());
    }

    // returns offset + sum(coord[d] * strides[d]) for a flat linear index
    size_t physical_offset(size_t linear_index) const {
        // linear index shouldn't be larger than number of elements
        if (linear_index >= static_cast<size_t>(numel())) {
            throw std::invalid_argument("View::physical_offset: linear index is out of bounds");
        }

        if (shape.empty()) {
            return static_cast<size_t>(offset);
        }

        // if tensor is already contiguous
        if (is_contiguous()) {
            return linear_index + static_cast<size_t>(offset);
        }

        // tensor isn't contiguous
        ViewVector dec(shape.size());

        // we traverse in reverse
        // decompose linear index first
        size_t tmp = linear_index;
        for (size_t i = shape.size(); i-- > 0;) {
            dec[i] = static_cast<ViewInt>(tmp % static_cast<size_t>(shape[i]));
            tmp /= static_cast<size_t>(shape[i]);
        }

        // then calculate physical offset
        ViewInt result = offset;
        for (size_t i = 0; i < dec.size(); ++i) {
            result += dec[i] * strides[i];
        }

        return static_cast<size_t>(result);
    }

    // --- Methods ---

    /*
     * - If shape is (a, b, c) and new_axis is (0, 2, 1),
     * then the new shape is (a, c, b).
     * - Reference: https://pytorch.org/docs/stable/generated/torch.permute.html
     */
    View permute(const ViewVector &new_axis) const {
        if (new_axis.size() != shape.size()) {
            throw std::invalid_argument("Permutation new axis doesn't match with current shape");
        }

        bool is_valid =
            std::all_of(new_axis.begin(), new_axis.end(), [&shape = shape](ViewInt axis) {
                return axis >= 0 && static_cast<size_t>(axis) < shape.size();
            });
        if (!is_valid) {
            throw std::invalid_argument("Invalid permute axis for current shape");
        }

        std::vector<bool> seen(shape.size(), false);
        for (ViewInt axis : new_axis) {
            auto idx = static_cast<size_t>(axis);
            if (seen[idx]) {
                throw std::invalid_argument("Permutation axes must be unique");
            }
            seen[idx] = true;
        }

        // create a new shape with permutation index
        ViewVector new_shape(shape.size());
        ViewVector new_strides(strides.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            new_shape[i] = shape[static_cast<size_t>(new_axis[i])];
            new_strides[i] = strides[static_cast<size_t>(new_axis[i])];
        }
        return {new_shape, new_strides, offset};
    }

    View transpose(ViewInt a, ViewInt b) const {
        if (a < 0 || b < 0 || static_cast<size_t>(a) >= shape.size() ||
            static_cast<size_t>(b) >= shape.size()) {
            throw std::invalid_argument("Transpose axes are out of bounds");
        }

        ViewVector axes(shape.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            axes[i] = static_cast<ViewInt>(i);
        }
        // then transpose the a and b from that axis
        std::swap(axes[static_cast<size_t>(a)], axes[static_cast<size_t>(b)]);

        return permute(axes);
    }

    /**
     * Shrink each axis to [start, end)
     *
     * - For example: `auto y = x.shrink({{1, 3}, {1, 3}});`
     * - Then: axis 0 keeps rows [1, 3), and axis 1 keeps cols [1, 3).
     */
    View shrink(const PairViewVector &limits) const {
        if (limits.size() != shape.size()) {
            throw std::invalid_argument("The numbers of each limit should match with shape size");
        }

        bool is_valid_limits = true;
        for (size_t i = 0; i < limits.size(); ++i) {
            is_valid_limits &= (limits[i].first < limits[i].second)               // start < end
                               && (limits[i].first >= 0 && limits[i].second >= 0) // non-negative
                               && (limits[i].second <= shape[i]); // end <= shape (in bound)
        }

        if (!is_valid_limits) {
            throw std::invalid_argument("limits is not valid");
        }

        ViewVector new_shape(shape.size());
        ViewInt new_offset = offset;
        for (size_t i = 0; i < shape.size(); ++i) {
            new_shape[i] = limits[i].second - limits[i].first;
            new_offset += limits[i].first * strides[i];
        }
        return {new_shape, strides, new_offset};
    }

    View expand(const ViewVector &new_shape) const {
        ViewInt new_ndim = static_cast<ViewInt>(new_shape.size());
        ViewInt current_ndim = static_cast<ViewInt>(dim()); // so the loop index can be negative

        if (new_ndim < current_ndim) {
            throw std::invalid_argument("expand shape cannot have fewer dimensions");
        }

        /*
         * What about expand in leading dimensions
         * (3, 2) -> (3, 3, 2) -> this is valid!
         * (3, 2) -> (5, 4, 3, 3, 2) -> this is valid too!
         * So the first thing we do we must increase dimension of current shape
         * (3, 2) -> (1, 3, 2) to match with (3, 3, 2)
         */
        ViewVector new_strides(static_cast<size_t>(new_ndim), 0);
        for (auto [i, j] = std::tuple(new_ndim - 1, current_ndim - 1); i >= 0; --i, --j) {
            ViewInt target_dim_size = new_shape[static_cast<size_t>(i)];
            ViewInt current_dim_size = (j >= 0) ? shape[static_cast<size_t>(j)] : 1;

            if (target_dim_size == current_dim_size) {
                new_strides[static_cast<size_t>(i)] =
                    (j >= 0) ? strides[static_cast<size_t>(j)] : 1;
            } else if (current_dim_size == 1) {
                new_strides[static_cast<size_t>(i)] = 0;
            } else {
                throw std::invalid_argument("expand shape is not valid for current shape");
            }
        }

        return {new_shape, new_strides, offset};
    }

    View reshape(const ViewVector &new_shape) const {
        ViewInt new_numel =
            std::reduce(new_shape.begin(), new_shape.end(), ViewInt{1}, std::multiplies<>());

        if (new_numel != numel()) {
            throw std::invalid_argument(
                "New shape cannot have different number of elements than current shape");
        }

        if (!is_contiguous()) {
            throw std::invalid_argument("Non-contiguous view cannot be reshaped without copy yet");
        }

        return {new_shape, offset};
    }
};
} // namespace comtam::core
