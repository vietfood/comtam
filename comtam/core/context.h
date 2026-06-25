#pragma once

#include "comtam/core/device.h"
#include "comtam/core/kernel.h"
#include <memory>

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
}
