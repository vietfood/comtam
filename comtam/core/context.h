/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include <memory>

#include "comtam/core/device.h"
#include "comtam/core/kernel.h"

namespace comtam::core {
class runtime_state {
   public:
    runtime_state();
    ~runtime_state() = default;

    runtime_state(const runtime_state&) = delete;
    runtime_state& operator=(const runtime_state&) = delete;

    metal_device& device() noexcept { return *device_; }
    const metal_device& device() const noexcept { return *device_; }

    kernel_library& kernels() noexcept { return *kernels_; }
    const kernel_library& kernels() const noexcept { return *kernels_; }

   private:
    std::unique_ptr<metal_device> device_;
    std::unique_ptr<kernel_library> kernels_;
};

class context {
   public:
    context();
    ~context() = default;

    // we want to copy a context so we can share the same runtime state
    context(const context&) = default;
    context& operator=(const context&) = default;

    // get method
    metal_device& device() noexcept { return state_->device(); }
    const metal_device& device() const noexcept { return state_->device(); }

    kernel_library& kernels() noexcept { return state_->kernels(); }
    const kernel_library& kernels() const noexcept { return state_->kernels(); }

    const std::shared_ptr<runtime_state>& state() const noexcept { return state_; }

    bool shares_runtime_with(const context& other) const noexcept { return state_ == other.state_; }

   private:
    std::shared_ptr<runtime_state> state_;
};

COMTAM_INLINE context& default_context() {
    static context instance;
    return instance;
}
}  // namespace comtam::core
