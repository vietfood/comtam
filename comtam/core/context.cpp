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
