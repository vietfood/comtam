#pragma once

#include <cstddef>
#include <unordered_map>

#include "comtam/macros/log.h"

namespace comtam::core {
enum class DType { Float32 };

static std::unordered_map<DType, size_t> dtype_size_map = {{DType::Float32, sizeof(float)}};

// clang-format off
#define COMTAM_DISPATCH_DTYPE(DTYPE, ...)                                                         \
    [&] {                                                                                         \
        switch (DTYPE) {                                                                          \
        case comtam::core::DType::Float32: {                                                      \
            using scalar_t = float;                                                               \
            return __VA_ARGS__();                                                                 \
        }                                                                                         \
        default:                                                                                  \
            COMTAM_THROW_ERROR(std::runtime_error, "unsupported dtype");                          \
        }                                                                                         \
    }()
// clang-format on

}  // namespace comtam::core
