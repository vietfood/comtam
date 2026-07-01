#include "comtam/core/storage.h"

#include <string>

#include "Metal/MTLDevice.hpp"
#include "Metal/MTLResource.hpp"
#include "comtam/macros/log.h"
#include "comtam/utils/common.h"

using namespace comtam::core;

Storage::Storage(size_t bytes, MTL::Device* device) : size_(bytes) {
    buffer_ = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));

    if (!buffer_) {
        COMTAM_LOG_ERR("Failed to allocate buffer of size {}", bytes);
        throw std::bad_alloc();
    }
}

void Storage::print(const std::string& label) const {
    comtam::utils::print_array(static_cast<float*>(buffer_->contents()), size_ / sizeof(float),
                               label);
}
