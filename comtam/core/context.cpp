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

#include "comtam/core/context.h"

#include <memory>

#ifndef COMTAM_KERNEL_DIR
#define COMTAM_KERNEL_DIR "kernels"
#endif

using namespace comtam::core;

runtime_state::runtime_state() {
    device_ = std::make_unique<metal_device>();
    kernels_ = std::make_unique<kernel_library>(device_->get(), COMTAM_KERNEL_DIR);
}

context::context() : state_(std::make_shared<runtime_state>()) {}