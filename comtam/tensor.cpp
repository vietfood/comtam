#include "comtam/tensor.h"

#include "comtam/core/kernel.h"
#include "comtam/core/view.h"

using namespace comtam;

// ----- Binary operation -----
Tensor Tensor::add(const Tensor &a, const Tensor &b, core::Device &device,
                   core::KernelLibrary &kernels) {
    // create an empty tensor
    Tensor out(a.view_, device);

    // first we create a command
    core::Command cmd = {
        .op = core::Op::ADD,
        .a = a.storage_.get(),
        .b = b.storage_.get(),
        .out = out.storage_.get(),
        .elements = a.numel(),
    };

    // then we submit the command
    device.submit(cmd, kernels);

    return out;
}

// ----- View operation -----
// These operations are free (change view only)

Tensor Tensor::permute(const std::vector<int64_t> &new_axis) {
    return Tensor(storage_, view_.permute(new_axis), dtype_);
}

Tensor Tensor::transpose(int64_t a, int64_t b) {
    return Tensor(storage_, view_.transpose(a, b), dtype_);
}

Tensor Tensor::shrink(const std::vector<std::pair<int64_t, int64_t>> &limits) {
    return Tensor(storage_, view_.shrink(limits), dtype_);
}

Tensor Tensor::expand(const std::vector<int64_t> &new_shape) {
    return Tensor(storage_, view_.expand(new_shape), dtype_);
}

Tensor Tensor::reshape(const std::vector<int64_t> &new_shape) {
    return Tensor(storage_, view_.reshape(new_shape), dtype_);
}
