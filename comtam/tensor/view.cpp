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

#include "comtam/tensor/view.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>

#include "comtam/macros/log.h"

using namespace comtam;

view::view(const view_vector& shape, view_int offset)
    : shape(shape), strides(shape.size(), 1), offset(offset), ref_strides(shape.size(), 1) {
    for (view_int i = static_cast<view_int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
        ref_strides[i] = ref_strides[i + 1] * shape[i + 1];
    }
}

view::view(const view_vector& shape, const view_vector& strides, view_int offset)
    : shape(shape), strides(strides), offset(offset), ref_strides(shape.size(), 1) {
    COMTAM_CHECK_AND_THROW(shape.size() == strides.size(), std::runtime_error,
                           "View: shape and strides must have the same rank");

    for (view_int i = static_cast<view_int>(shape.size()) - 2; i >= 0; --i) {
        ref_strides[i] = ref_strides[i + 1] * shape[i + 1];
    }
}

view_int view::numel() const {
    return std::reduce(shape.begin(), shape.end(), view_int{1}, std::multiplies<>());
}

size_int view::physical_offset(view_int linear_index) const {
    // linear index shouldn't be larger than number of elements
    COMTAM_CHECK_AND_THROW(linear_index < numel(), std::invalid_argument,
                           "View::physical_offset: linear index is out of bounds");

    if (shape.empty()) {
        return static_cast<size_int>(offset);
    }

    // if tensor is already contiguous
    if (is_contiguous()) {
        return static_cast<size_int>(linear_index + offset);
    }

    // tensor isn't contiguous
    view_vector dec(shape.size());

    /* we traverse in reverse: decompose linear index first */
    view_int tmp = linear_index;
    for (view_int i = static_cast<view_int>(shape.size()) - 1; i >= 0; --i) {
        dec[static_cast<size_t>(i)] = tmp % shape[static_cast<size_t>(i)];
        tmp /= shape[static_cast<size_t>(i)];
    }

    // then calculate physical offset
    view_int result = offset;
    for (size_t i = 0; i < dec.size(); ++i) {
        result += dec[i] * strides[i];
    }

    return static_cast<size_int>(result);
}

view view::permute(const view_vector& new_axis) const {
    COMTAM_CHECK_AND_THROW(new_axis.size() == shape.size(), std::invalid_argument,
                           "Permutation new axis doesn't match with current shape");

    bool is_valid = std::all_of(new_axis.begin(), new_axis.end(), [&shape = shape](view_int axis) {
        return axis >= 0 && static_cast<size_t>(axis) < shape.size();
    });
    COMTAM_CHECK_AND_THROW(is_valid, std::invalid_argument,
                           "Invalid permute axis for current shape");

    std::vector<bool> seen(shape.size(), false);
    for (view_int axis : new_axis) {
        auto idx = static_cast<size_t>(axis);
        COMTAM_CHECK_AND_THROW(!seen[idx], std::invalid_argument,
                               "Permutation axes must be unique");
        seen[idx] = true;
    }

    // create a new shape with permutation index
    view_vector new_shape(shape.size());
    view_vector new_strides(strides.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        new_shape[i] = shape[static_cast<size_t>(new_axis[i])];
        new_strides[i] = strides[static_cast<size_t>(new_axis[i])];
    }
    return {new_shape, new_strides, offset};
}

view view::transpose(view_int a, view_int b) const {
    COMTAM_CHECK_AND_THROW(a >= 0 && b >= 0 && static_cast<size_t>(a) < shape.size() &&
                               static_cast<size_t>(b) < shape.size(),
                           std::invalid_argument, "Transpose axes are out of bounds");

    view_vector axes(shape.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        axes[i] = static_cast<view_int>(i);
    }
    // then transpose the a and b from that axis
    std::swap(axes[static_cast<size_t>(a)], axes[static_cast<size_t>(b)]);

    return permute(axes);
}

view view::shrink(const pair_view_vector& limits) const {
    COMTAM_CHECK_AND_THROW(limits.size() == shape.size(), std::invalid_argument,
                           "The numbers of each limit should match with shape size");

    bool is_valid_limits = true;
    for (size_t i = 0; i < limits.size(); ++i) {
        is_valid_limits &= (limits[i].first < limits[i].second)                // start < end
                           && (limits[i].first >= 0 && limits[i].second >= 0)  // non-negative
                           && (limits[i].second <= shape[i]);  // end <= shape (in bound)
    }

    COMTAM_CHECK_AND_THROW(is_valid_limits, std::invalid_argument, "limits is not valid");

    view_vector new_shape(shape.size());
    view_int new_offset = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
        new_shape[i] = limits[i].second - limits[i].first;
        new_offset += limits[i].first * strides[i];
    }
    return {new_shape, strides, new_offset};
}

view view::expand(const view_vector& new_shape) const {
    view_int new_ndim = static_cast<view_int>(new_shape.size());
    view_int current_ndim = dim();  // so the loop index can be negative

    COMTAM_CHECK_AND_THROW(new_ndim >= current_ndim, std::invalid_argument,
                           "expand shape cannot have fewer dimensions");

    /*
     * What about expand in leading dimensions
     * (3, 2) -> (3, 3, 2) -> this is valid!
     * (3, 2) -> (5, 4, 3, 3, 2) -> this is valid too!
     * So the first thing we do we must increase dimension of current shape
     * (3, 2) -> (1, 3, 2) to match with (3, 3, 2)
     */
    view_vector new_strides(static_cast<size_t>(new_ndim), 0);
    for (auto [i, j] = std::tuple(new_ndim - 1, current_ndim - 1); i >= 0; --i, --j) {
        view_int target_dim_size = new_shape[static_cast<size_t>(i)];
        view_int current_dim_size = (j >= 0) ? shape[static_cast<size_t>(j)] : 1;

        if (target_dim_size == current_dim_size) {
            new_strides[static_cast<size_t>(i)] = (j >= 0) ? strides[static_cast<size_t>(j)] : 1;
        } else if (current_dim_size == 1) {
            new_strides[static_cast<size_t>(i)] = 0;
        } else {
            COMTAM_THROW_ERROR(std::invalid_argument,
                               "expand shape is not valid for current shape");
        }
    }

    return {new_shape, new_strides, offset};
}

view view::reshape(const view_vector& new_shape) const {
    view_int new_numel =
        std::reduce(new_shape.begin(), new_shape.end(), view_int{1}, std::multiplies<>());

    COMTAM_CHECK_AND_THROW(new_numel == numel(), std::invalid_argument,
                           "New shape cannot have different number of elements than current shape");

    COMTAM_CHECK_AND_THROW(is_contiguous(), std::invalid_argument,
                           "Non-contiguous view cannot be reshaped without copy yet");

    return {new_shape, offset};
}

view_vector view::broadcast_shape(const view& lhs, const view& rhs) {
    view_vector big = (lhs.dim() >= rhs.dim()) ? lhs.shape : rhs.shape;
    view_vector small = (lhs.dim() >= rhs.dim()) ? rhs.shape : lhs.shape;

    const view_int ndim = static_cast<view_int>(big.size());
    const view_int offset =
        ndim - static_cast<view_int>(small.size());  // small is left-padded with 1s

    view_vector new_shape(static_cast<size_t>(ndim));

    for (view_int i = ndim - 1; i >= 0; --i) {
        view_int a = big[static_cast<size_t>(i)];
        view_int b = (i >= offset) ? small[static_cast<size_t>(i - offset)] : 1;

        if (a != b && a != 1 && b != 1) {
            COMTAM_THROW_ERROR(std::runtime_error, "Both shapes aren't compatible for broadcast");
        }

        if (a == 1 || b == 1) {
            new_shape[static_cast<size_t>(i)] = a * b;
        } else {  // a == b
            new_shape[static_cast<size_t>(i)] = a;
        }
    }

    return new_shape;
}
