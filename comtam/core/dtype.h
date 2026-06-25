#pragma once

#include <unordered_map>
#include <cstddef>

namespace comtam::core {
enum class DType {
    Float32
};

static std::unordered_map<DType, size_t> dtype_size_map = {
    {DType::Float32, sizeof(float)}
};
}
