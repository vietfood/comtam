#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace comtam::core {
using ViewInt = int64_t;

struct View {
    std::vector<ViewInt> shape;
    std::vector<ViewInt> strides;
    // for contiguous check
    std::vector<ViewInt> ref_strides;
    size_t offset = 0;

    View(std::vector<ViewInt> shape)
        : shape(shape), strides(shape.size(), 1), ref_strides(shape.size(), 1) {
        for (ViewInt i = static_cast<ViewInt>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
            ref_strides[i] = ref_strides[i + 1] * shape[i + 1];
        }
    }

    bool is_contiguous() const {
        return strides == ref_strides;
    }

    size_t dim() const {
        return shape.size();
    }

    size_t numel() const {
        size_t result = 1;
        for (size_t s : shape) {
            result *= s;
        }
        return result;
    }
};
}
