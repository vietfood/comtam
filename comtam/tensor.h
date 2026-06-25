#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "comtam/core/dtype.h"
#include "comtam/core/view.h"

namespace comtam {
namespace core {
    class Storage;
    class Device;
}

class Tensor {
public:
    Tensor(const float* data, std::vector<core::ViewInt> shape, core::Device& device);
    Tensor(const std::vector<core::ViewInt>& shape, core::Device& device);

    void from_vector(const std::vector<float>& data, core::Device& device);
    std::vector<float> to_vector(core::Device& device) const;

    size_t numel() const {
        return view_.numel();
    }

    size_t dim() const {
        return view_.dim();
    }

private:
    core::DType dtype_;
    core::View view_;
    std::shared_ptr<core::Storage> storage_;
};
}
