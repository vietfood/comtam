#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLDevice.hpp"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"

namespace comtam::core {
struct Command;
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

    template <typename T>
    void copy(const T* data, size_t count, Storage& storage) {
        const size_t bytes = count * sizeof(T);
        COMTAM_CHECK_AND_THROW(bytes == storage.size(), std::runtime_error,
                               "Data size does not match storage size");
        std::memcpy(storage.ptr()->contents(), data, bytes);
        storage.ptr()->didModifyRange(NS::Range(0, bytes));
    }

    template <typename T>
    void copy(Storage& storage, T* data, size_t count) {
        const size_t bytes = count * sizeof(T);
        COMTAM_CHECK_AND_THROW(bytes == storage.size(), std::runtime_error,
                               "Data size does not match storage size");
        std::memcpy(data, storage.ptr()->contents(), bytes);
    }

    void copy(Storage& src, Storage& dst) {
        COMTAM_CHECK_AND_THROW(src.size() == dst.size(), std::runtime_error,
                               "Storage sizes do not match");
        std::memcpy(dst.ptr()->contents(), src.ptr()->contents(), src.size());
        dst.ptr()->didModifyRange(NS::Range(0, dst.size()));
    }

    // methods for command execution
    void submit(const Command& command, KernelLibrary& kernels);

   private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> command_queue_;
};
}  // namespace comtam::core
