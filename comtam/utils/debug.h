#pragma once

#include <cstdint>
#include <string>

#include <fmt/format.h>

#include "comtam/core/ops.h"
#include "comtam/core/view.h"
#include "comtam/macros/macros.h"

namespace comtam::utils {
COMTAM_INLINE std::string format_arr4(const int64_t arr[4]) {
    return fmt::format("({}, {}, {}, {})", arr[0], arr[1], arr[2], arr[3]);
}

COMTAM_INLINE std::string format_view_vector(const core::ViewVector& vec) {
    std::string out = "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        out += fmt::format("{}{}", vec[i], (i + 1 < vec.size()) ? ", " : "");
    }
    out += "]";
    return out;
}

COMTAM_INLINE std::string format_view(const core::View& view) {
    return fmt::format("shape={}, strides={}, offset={}, contiguous={}",
                       format_view_vector(view.shape), format_view_vector(view.strides),
                       view.offset, view.is_contiguous());
}

COMTAM_INLINE std::string format_view_info(const core::ViewInfo& info) {
    return fmt::format("N={}, shape={}, strides={}, offset={}, contiguous={}", info.N,
                       format_arr4(info.shape), format_arr4(info.strides), info.offset,
                       info.contiguous);
}
}  // namespace comtam::utils
