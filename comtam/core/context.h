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
class context {
   public:
    context();
    ~context() = default;

    // get method
    metal_device& device() { return *device_; }
    const metal_device& device() const { return *device_; }
    kernel_library& kernels() { return *kernels_; }
    const kernel_library& kernels() const { return *kernels_; }

   private:
    std::unique_ptr<metal_device> device_;
    std::unique_ptr<kernel_library> kernels_;
};
}  // namespace comtam::core
