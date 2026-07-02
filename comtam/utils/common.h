#pragma once

#include <simd/vector_make.h>
#include <simd/vector_types.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "Foundation/NSError.hpp"
#include "comtam/macros/macros.h"

namespace comtam::utils {
COMTAM_INLINE std::string ns_error_message(NS::Error* error) {
    if (error == nullptr) {
        return "Unknown Metal error";
    }

    NS::String* description = error->localizedDescription();
    if (description == nullptr) {
        return "Unknown Metal error";
    }

    return description->utf8String();
}

// a small read file util
// https://stackoverflow.com/questions/6755111/read-input-files-fastest-way-possible
COMTAM_INLINE std::string read_file(const std::string& path) {
    // Open in binary mode to avoid conversion overhead
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return "";

    // Determine file size and pre-allocate memory
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string buffer;
    buffer.resize(size);

    // Read everything in one single operation
    if (file.read(&buffer[0], size)) {
        return buffer;
    }
    return "";
}

template <typename T>
COMTAM_INLINE void print_array(T* arr, std::size_t size, std::string_view label = "") {
    if (!label.empty()) {
        fmt::print("{}: [", label);
    } else {
        fmt::print("[");
    }
    for (std::size_t i = 0; i < size; ++i) {
        fmt::print("{}{}", arr[i], (i + 1 < size) ? ", " : "");
    }
    fmt::print("]\n");
}

COMTAM_INLINE std::array<int64_t, 4> arr4_from_vec(const std::vector<int64_t>& vec) {
    std::array<int64_t, 4> temp = {-1, -1, -1, -1};
    std::copy_n(vec.begin(), std::min(vec.size(), size_t(4)), temp.begin());
    return temp;
}
}  // namespace comtam::utils
