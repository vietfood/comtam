#include "comtam/tensor.h"

#include <stdexcept>

#include "comtam/core/context.h"
#include "comtam/core/ops.h"
#include "comtam/core/view.h"
#include "comtam/macros/log.h"
#include "comtam/utils/debug.h"

using namespace comtam;

// ----- Binary operation -----
Tensor Tensor::bop(const Tensor& a, const Tensor& b, const core::Op& op, core::Context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    COMTAM_CHECK_AND_THROW(a.dtype_ == b.dtype_, std::runtime_error,
                           "Two operands must have the same dtype");

    COMTAM_CHECK_AND_THROW(a.view_.is_contiguous() && b.view_.is_contiguous(), std::runtime_error,
                           "Cannot do operation on non-contiguous inputs");

    COMTAM_LOG_DEBUG("binop {}_{}:\na={}\nb={}\n", core::op2kernel(op),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

    auto final_shape = core::View::broadcast_shape(a.view_, b.view_);

    Tensor a_expand = a.expand(final_shape);
    Tensor b_expand = b.expand(final_shape);

    COMTAM_LOG_DEBUG("binop broadcast: shape={}\na_expand={}\nb_expand={}\n",
                     utils::format_view_vector(final_shape), utils::format_view(a_expand.view_),
                     utils::format_view(b_expand.view_));

    Tensor out(final_shape, device, a.dtype_);

    core::Command cmd = {.kernel = {.op = op, .dtype = a.dtype_},
                         .a = {.storage = a_expand.storage_.get(),
                               .view = core::ViewInfo::from_view(a_expand.view_)},
                         .b = {.storage = b_expand.storage_.get(),
                               .view = core::ViewInfo::from_view(b_expand.view_)},
                         .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("binop ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    device.submit_bop(cmd, kernels);

    return out;
}

Tensor Tensor::add(const Tensor& a, const Tensor& b, core::Context& ctx) {
    return Tensor::bop(a, b, core::Op::ADD, ctx);
}

Tensor Tensor::sub(const Tensor& a, const Tensor& b, core::Context& ctx) {
    return Tensor::bop(a, b, core::Op::SUB, ctx);
}

Tensor Tensor::mul(const Tensor& a, const Tensor& b, core::Context& ctx) {
    return Tensor::bop(a, b, core::Op::MUL, ctx);
}

Tensor Tensor::div(const Tensor& a, const Tensor& b, core::Context& ctx) {
    return Tensor::bop(a, b, core::Op::DIV, ctx);
}

// ----- Matmul -----
Tensor Tensor::matmul(const Tensor& a, const Tensor& b, core::Context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    COMTAM_CHECK_AND_THROW(a.dtype_ == b.dtype_, std::runtime_error,
                           "Two operands must have the same dtype");

    COMTAM_CHECK_AND_THROW(a.shape().size() == 2 && b.shape().size() == 2, std::runtime_error,
                           "Right now, matmul only supports 2D array");

    COMTAM_CHECK_AND_THROW(a.shape()[1] == b.shape()[0], std::runtime_error,
                           "To do matmul, column of a must match with row of b");

    COMTAM_LOG_DEBUG("matmul {}_{}:\na={}\nb={}\n", core::op2kernel(core::Op::MATMUL),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

    Tensor out({a.shape()[0], b.shape()[1]}, device, a.dtype_);

    core::Command cmd = {.kernel = {.op = core::Op::MATMUL, .dtype = a.dtype_},
                         .a = {.storage = a.storage_.get(),
                               .view = core::ViewInfo::from_view(a.view_)},
                         .b = {.storage = b.storage_.get(),
                               .view = core::ViewInfo::from_view(b.view_)},
                         .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("matmul ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    device.submit_matmul(cmd, kernels);

    return out;
}

// ----- View operation -----
// These operations are free (change view only)

Tensor Tensor::permute(const std::vector<int64_t>& new_axis) const {
    return Tensor(storage_, view_.permute(new_axis), dtype_);
}

Tensor Tensor::transpose(int64_t a, int64_t b) const {
    return Tensor(storage_, view_.transpose(a, b), dtype_);
}

Tensor Tensor::shrink(const std::vector<std::pair<int64_t, int64_t>>& limits) const {
    return Tensor(storage_, view_.shrink(limits), dtype_);
}

Tensor Tensor::expand(const std::vector<int64_t>& new_shape) const {
    return Tensor(storage_, view_.expand(new_shape), dtype_);
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    return Tensor(storage_, view_.reshape(new_shape), dtype_);
}
