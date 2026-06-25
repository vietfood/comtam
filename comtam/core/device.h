#pragma once

#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLDevice.hpp"


namespace comtam::core {
class Storage;

class Device {
public:
    Device();

    // NS::SharedPtr will automatically handle memory management
    ~Device() = default;

    // get method
    const MTL::Device* get() const { return device_.get(); }
    MTL::Device* get() { return device_.get(); }

    const MTL::CommandQueue* queue() const { return command_queue_.get(); }
    MTL::CommandQueue* queue() { return command_queue_.get(); }

    // other methods
    Storage allocate(size_t bytes);
    void copy(const std::vector<float>& data, Storage& storage);
    void copy(Storage& storage, std::vector<float>& data);

private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> command_queue_;
};
}
