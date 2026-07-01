#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "Foundation/NSError.hpp"

namespace comtam::utils {
inline std::string ns_error_message(NS::Error* error) {
    if (error == nullptr) {
        return "[ERROR] Unknown Metal error";
    }

    NS::String* description = error->localizedDescription();
    if (description == nullptr) {
        return "[ERROR] Unknown Metal error";
    }

    return description->utf8String();
}

// a small read file util
// https://stackoverflow.com/questions/6755111/read-input-files-fastest-way-possible
inline std::string read_file(const std::string& path) {
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

// pretty-print array
template <typename T>
void print_array(T* arr, std::size_t size, std::string_view label = "") {
    if (!label.empty()) {
        std::cout << label << ": ";
    }
    std::cout << "[";
    for (std::size_t i = 0; i < size; ++i) {
        std::cout << arr[i];
        if (i + 1 < size) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}
}  // namespace comtam::utils
