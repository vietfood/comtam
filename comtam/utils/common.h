/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include <fmt/format.h>
#include <simd/vector_make.h>
#include <simd/vector_types.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <vector>

#include "Foundation/NSError.hpp"
#include "comtam/macros/macros.h"
#include "comtam/types.h"

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

/* a small read file util:
 * https://stackoverflow.com/questions/6755111/read-input-files-fastest-way-possible */
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
COMTAM_INLINE void print_array(T* arr, size_int size, std::string_view label = "") {
    if (!label.empty()) {
        fmt::print("{}: [", label);
    } else {
        fmt::print("[");
    }
    for (size_int i = 0; i < size; ++i) {
        fmt::print("{}{}", arr[i], (i + 1 < size) ? ", " : "");
    }
    fmt::print("]\n");
}

/*
 * Pack a view shape/stride vector into a fixed 4-array, padding with -1
 * (the sentinel used by kernels/utils.h ViewInfo for undefined dims).
 */
COMTAM_INLINE std::array<view_int, 4> arr4_from_vec(const std::vector<view_int>& vec) {
    std::array<view_int, 4> temp = {-1, -1, -1, -1};
    std::copy_n(vec.begin(), std::min(vec.size(), size_int(4)), temp.begin());
    return temp;
}
}  // namespace comtam::utils
