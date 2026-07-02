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

#include <memory>

#include "comtam/core/device.h"
#include "comtam/core/kernel.h"

namespace comtam::core {
class Context {
   public:
    Context();
    ~Context() = default;

    // get method
    Device& device() { return *device_; }
    const Device& device() const { return *device_; }
    KernelLibrary& kernels() { return *kernels_; }
    const KernelLibrary& kernels() const { return *kernels_; }

   private:
    std::unique_ptr<Device> device_;
    std::unique_ptr<KernelLibrary> kernels_;
};
}  // namespace comtam::core
