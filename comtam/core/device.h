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
#include <cstring>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLCommandQueue.hpp"
#include "Metal/MTLDevice.hpp"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/types.h"

namespace comtam::core {
struct command_desc;
class kernel_library;

class metal_device {
   public:
    metal_device();

    // NS::SharedPtr will automatically handle memory management
    ~metal_device() = default;

    // get method
    const MTL::Device* get() const { return device_.get(); }
    MTL::Device* get() { return device_.get(); }

    const MTL::CommandQueue* queue() const { return command_queue_.get(); }
    MTL::CommandQueue* queue() { return command_queue_.get(); }

    // methods for storage
    storage allocate(size_int bytes);

    template <typename T>
    void copy(const T* data, size_int count, storage& storage) {
        const size_int bytes = count * sizeof(T);
        COMTAM_CHECK_AND_THROW(bytes == storage.size(), std::runtime_error,
                               "Data size does not match storage size");
        std::memcpy(storage.ptr()->contents(), data, bytes);
        storage.ptr()->didModifyRange(NS::Range(0, bytes));
    }

    template <typename T>
    void copy(T value, storage& storage) {
        COMTAM_CHECK_AND_THROW(sizeof(T) == storage.size(), std::runtime_error,
                               "Data size does not match storage size");
        std::memcpy(storage.ptr()->contents(), &value, sizeof(T));
        storage.ptr()->didModifyRange(NS::Range(0, sizeof(T)));
    }

    template <typename T>
    void copy(storage& storage, T* data, size_int count) {
        const size_int bytes = count * sizeof(T);
        COMTAM_CHECK_AND_THROW(bytes == storage.size(), std::runtime_error,
                               "Data size does not match storage size");
        std::memcpy(data, storage.ptr()->contents(), bytes);
    }

    void copy(storage& src, storage& dst) {
        COMTAM_CHECK_AND_THROW(src.size() == dst.size(), std::runtime_error,
                               "Storage sizes do not match");
        std::memcpy(dst.ptr()->contents(), src.ptr()->contents(), src.size());
        dst.ptr()->didModifyRange(NS::Range(0, dst.size()));
    }

    // methods for command execution
    void submit_bop(const command_desc& command, kernel_library& kernels);
    void submit_uop(const command_desc& command, kernel_library& kernels);
    void submit_matmul(const command_desc& command, kernel_library& kernels);
    void submit_reduce(const command_desc& command, kernel_library& kernels);

   private:
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> command_queue_;
};
}  // namespace comtam::core
