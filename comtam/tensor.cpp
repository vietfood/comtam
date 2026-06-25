#include "comtam/tensor.h"
#include "comtam/core/storage.h"
#include "comtam/core/device.h"

using namespace comtam;

Tensor::Tensor(const float* data, std::vector<core::ViewInt> shape, core::Device& device)
    : dtype_(core::DType::Float32),
    view_(shape),
    storage_(nullptr) {

    // create a storage from device first
    auto buffer = device.allocate(core::dtype_size_map[dtype_] * view_.numel());
    device.copy(data, view_.numel(), buffer);

    // then wrap to shared_ptr
    storage_ = std::make_shared<core::Storage>(std::move(buffer));
}

Tensor::Tensor(const std::vector<core::ViewInt>& shape, core::Device& device)
    : dtype_(core::DType::Float32),
    view_(shape),
    storage_(nullptr) {

    // we create an empty buffer
    auto buffer = device.allocate(core::dtype_size_map[dtype_] * view_.numel());
    storage_ = std::make_shared<core::Storage>(std::move(buffer));
}

void Tensor::from_vector(const std::vector<float>& data, core::Device& device) {
    // we assume that this tensor is allocated
    if (!storage_) {
        throw std::runtime_error("Tensor::from_vector: storage is not allocated");
    }

    if (data.size() != view_.numel()) {
        throw std::runtime_error("Tensor::from_vector: data size does not match tensor size");
    }

    device.copy(data.data(), data.size(), *storage_.get());
}

std::vector<float> Tensor::to_vector(core::Device& device) const {
    if (!storage_) {
        throw std::runtime_error("Tensor::from_vector: storage is not allocated");
    }

    std::vector<float> result(view_.numel());
    device.copy(*storage_.get(), result.data(), result.size());
    return result;
}
