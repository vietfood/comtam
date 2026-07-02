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

Context::Context() {
    device_ = std::make_unique<Device>();
    kernels_ = std::make_unique<KernelLibrary>(device_->get(), COMTAM_KERNEL_DIR);
}
