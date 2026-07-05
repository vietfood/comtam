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
#include <string>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLBuffer.hpp"
#include "comtam/macros/log.h"

namespace comtam::core {
class storage {
   public:
    storage(size_t bytes, MTL::Device* device);
    ~storage() = default;

    // move constructor
    storage(storage&& other) noexcept : size_(other.size_), buffer_(std::move(other.buffer_)) {}

    storage& operator=(storage&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            buffer_ = std::move(other.buffer_);
        }
        return *this;
    }

    // we want move only
    storage(const storage& other) = delete;
    storage& operator=(const storage& other) = delete;

    size_t size() const { return size_; }
    MTL::Buffer* ptr() { return buffer_.get(); }
    const MTL::Buffer* ptr() const { return buffer_.get(); }

    template <typename T>
    T at(size_t index) const {
        const size_t byte_offset = index * sizeof(T);
        COMTAM_CHECK_AND_THROW(byte_offset + sizeof(T) <= size_, std::runtime_error,
                               "Storage index out of bounds");
        return static_cast<const T*>(buffer_->contents())[index];
    }

    void print(const std::string& label) const;

   private:
    size_t size_;
    NS::SharedPtr<MTL::Buffer> buffer_;
};
}  // namespace comtam::core
