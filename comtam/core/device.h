#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLDevice.hpp"
#include "comtam/core/storage.h"


namespace comtam::core {
class Command;
class KernelLibrary;

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

    // methods for storage
    Storage allocate(size_t bytes);

    template<typename T>
    void copy(const T* data, size_t count, Storage& storage) {
        const size_t bytes = count * sizeof(T);
        if (bytes != storage.size()) {
            throw std::runtime_error("[ERROR] Data size does not match storage size");
        }
        std::memcpy(storage.ptr()->contents(), data, bytes);
        storage.ptr()->didModifyRange(NS::Range(0, bytes));
    }

    template<typename T>
    void copy(Storage& storage, T* data, size_t count) {
        const size_t bytes = count * sizeof(T);
        if (bytes != storage.size()) {
            throw std::runtime_error("[ERROR] Data size does not match storage size");
        }
        std::memcpy(data, storage.ptr()->contents(), bytes);
    }

    // methods for command execution
    void submit(const Command& command, KernelLibrary& kernels);

private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> command_queue_;
};
}
