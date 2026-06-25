#pragma once

#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLBuffer.hpp"
#include <cstddef>

namespace comtam::core {
class Storage {
public:
    Storage(size_t bytes, MTL::Device* device);
    ~Storage() = default;

    size_t size() const { return size_; }
    MTL::Buffer* ptr() { return buffer_.get(); }
    const MTL::Buffer* ptr() const { return buffer_.get(); }

    void print(const std::string& label) const;
private:
    size_t size_;
    NS::SharedPtr<MTL::Buffer> buffer_;
};
}
