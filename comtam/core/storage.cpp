#include "comtam/core/storage.h"

#include <iostream>
#include <string>

#include "Metal/MTLDevice.hpp"
#include "Metal/MTLResource.hpp"
#include "comtam/utils/common.h"

using namespace comtam::core;

Storage::Storage(size_t bytes, MTL::Device* device) : size_(bytes) {
    buffer_ = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));

    if (!buffer_) {
        std::cerr << "[ERROR] Failed to allocate buffer of size " << bytes << "\n";
        throw std::bad_alloc();
    }
}

void Storage::print(const std::string& label) const {
    comtam::utils::print_array(static_cast<float*>(buffer_->contents()), size_ / sizeof(float),
                               label);
}
