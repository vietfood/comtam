#include "comtam/core/device.h"
#include "comtam/core/storage.h"

using namespace comtam::core;

Device::Device() {
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device_) {
        throw std::runtime_error("[ERROR] Failed to create default Metal device");
    }

    command_queue_ = NS::TransferPtr(device_->newCommandQueue());
    if (!command_queue_) {
        throw std::runtime_error("[ERROR] Failed to create Metal command queue");
    }
}

Storage Device::allocate(size_t bytes) {
    return Storage(bytes, this->get());
}

void Device::copy(const std::vector<float>& data, Storage& storage) {
    if (data.size() * sizeof(float) != storage.size()) {
        throw std::runtime_error("[ERROR] Data size does not match storage size");
    }
    memcpy(storage.ptr()->contents(), data.data(), data.size() * sizeof(float));
    storage.ptr()->didModifyRange(NS::Range(0, data.size() * sizeof(float)));
}

// copy from Storage back to std::vector<float>
void Device::copy(Storage& storage, std::vector<float>& data) {
    if (data.size() * sizeof(float) != storage.size()) {
        throw std::runtime_error("[ERROR] Data size does not match storage size");
    }
    data.resize(storage.size());
    memcpy(data.data(), storage.ptr()->contents(), data.size() * sizeof(float));
}
