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

#include "comtam/macros/log.h"
#include "comtam/macros/macros.h"

namespace comtam {
enum class DType { Float32 };

COMTAM_INLINE bool support_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return true;
        default:
            return false;
    }
}

// clang-format off
#define COMTAM_DISPATCH_DTYPE(DTYPE, ...)                                                         \
    [&] {                                                                                         \
        switch (DTYPE) {                                                                          \
        case comtam::DType::Float32: {                                                            \
            using scalar_t = float;                                                               \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        default:                                                                                  \
            COMTAM_THROW_ERROR(std::runtime_error, "unsupported dtype");                          \
        }                                                                                         \
    }()
// clang-format on

}  // namespace comtam
