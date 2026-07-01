#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLBuffer.hpp"

namespace comtam::core {
class Storage {
   public:
    Storage(size_t bytes, MTL::Device* device);
    ~Storage() = default;

    // move constructor
    Storage(Storage&& other) noexcept : size_(other.size_), buffer_(std::move(other.buffer_)) {}

    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            buffer_ = std::move(other.buffer_);
        }
        return *this;
    }

    // we want move only
    Storage(const Storage& other) = delete;
    Storage& operator=(const Storage& other) = delete;

    size_t size() const { return size_; }
    MTL::Buffer* ptr() { return buffer_.get(); }
    const MTL::Buffer* ptr() const { return buffer_.get(); }

    template <typename T>
    T at(size_t index) const {
        const size_t byte_offset = index * sizeof(T);
        if (byte_offset + sizeof(T) > size_) {
            throw std::runtime_error("[ERROR] Storage index out of bounds");
        }
        return static_cast<const T*>(buffer_->contents())[index];
    }

    void print(const std::string& label) const;

   private:
    size_t size_;
    NS::SharedPtr<MTL::Buffer> buffer_;
};
}  // namespace comtam::core
