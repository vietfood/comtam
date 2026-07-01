#include "comtam/tensor.h"

#include "comtam/core/context.h"
#include "comtam/core/ops.h"
#include "comtam/core/view.h"
#include <stdexcept>

using namespace comtam;

// ----- Binary operation -----
Tensor Tensor::binop(const Tensor &a, const Tensor &b, const core::Op& op, core::Context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    if (a.dtype_ != b.dtype_) {
        throw std::runtime_error("Two operands must have the same dtype");
    }

    if (!a.view_.is_contiguous() || !b.view_.is_contiguous()) {
        throw std::runtime_error("Cannot do operation on non-contiguous inputs");
    }

    if (a.view_ != b.view_) {
        throw std::runtime_error("Two operands must have the same shape");
    }

    // create an empty tensor
    Tensor out(a.view_, device, a.dtype_);

    // first we create a command
    core::Command cmd = {
        .kernel = {.op = op, .dtype = a.dtype_},
        .a = a.storage_.get(),
        .b = b.storage_.get(),
        .out = out.storage_.get(),
        .elements = a.numel(),
    };

    // then we submit the command
    device.submit(cmd, kernels);

    return out;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::ADD, ctx);
}

Tensor Tensor::sub(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::SUB, ctx);
}

Tensor Tensor::mul(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::MUL, ctx);
}

Tensor Tensor::div(const Tensor &a, const Tensor &b, core::Context& ctx) {
    return Tensor::binop(a, b, core::Op::DIV, ctx);
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
