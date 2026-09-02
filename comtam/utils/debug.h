/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** | AI assist : Composer 2.5 (Cursor) and GPT 5.5 (Codex)                 |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include <fmt/format.h>

#include <cstdint>
#include <string>

#include "comtam/core/command.h"
#include "comtam/macros/macros.h"
#include "comtam/tensor/view.h"

namespace comtam::utils {
COMTAM_INLINE std::string format_arr4(const view_int arr[4]) {
    return fmt::format("({}, {}, {}, {})", arr[0], arr[1], arr[2], arr[3]);
}

COMTAM_INLINE std::string format_view_vector(const view_vector& vec) {
    std::string out = "[";
    for (size_int i = 0; i < vec.size(); ++i) {
        out += fmt::format("{}{}", vec[i], (i + 1 < vec.size()) ? ", " : "");
    }
    out += "]";
    return out;
}

COMTAM_INLINE std::string format_view(const view& view) {
    return fmt::format("shape={}, strides={}, offset={}, contiguous={}",
                       format_view_vector(view.shape), format_view_vector(view.strides),
                       view.offset, view.is_contiguous());
}

COMTAM_INLINE std::string format_view_info(const core::view_desc& info) {
    return fmt::format("N={}, shape={}, strides={}, offset={}, contiguous={}", info.N,
                       format_arr4(info.shape), format_arr4(info.strides), info.offset,
                       info.contiguous);
}
}  // namespace comtam::utils
