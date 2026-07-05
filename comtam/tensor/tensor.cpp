#include "comtam/tensor/tensor.h"

#include <stdexcept>

#include "comtam/core/command.h"
#include "comtam/core/context.h"
#include "comtam/macros/log.h"
#include "comtam/utils/debug.h"

using namespace comtam;

// ----- Binary operation -----
tensor tensor::bop(const tensor& a, const tensor& b, const Op& op, core::context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    COMTAM_CHECK_AND_THROW(a.dtype_ == b.dtype_, std::runtime_error,
                           "Two operands must have the same dtype");

    COMTAM_CHECK_AND_THROW(a.view_.is_contiguous() && b.view_.is_contiguous(), std::runtime_error,
                           "Cannot do operation on non-contiguous inputs");

    COMTAM_LOG_DEBUG("binop {}_{}:\na={}\nb={}\n", core::op2kernel(op),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

    auto final_shape = view::broadcast_shape(a.view_, b.view_);

    tensor a_expand = a.expand(final_shape);
    tensor b_expand = b.expand(final_shape);

    COMTAM_LOG_DEBUG("binop broadcast: shape={}\na_expand={}\nb_expand={}\n",
                     utils::format_view_vector(final_shape), utils::format_view(a_expand.view_),
                     utils::format_view(b_expand.view_));

    tensor out(final_shape, device, a.dtype_);

    core::command_desc cmd = {.kernel = {.op = op, .dtype = a.dtype_},
                              .a = {.storage = a_expand.storage_.get(),
                                    .view = core::view_desc::from_view(a_expand.view_)},
                              .b = {.storage = b_expand.storage_.get(),
                                    .view = core::view_desc::from_view(b_expand.view_)},
                              .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("binop ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    device.submit_bop(cmd, kernels);

    return out;
}

tensor tensor::add(const tensor& a, const tensor& b, core::context& ctx) {
    return tensor::bop(a, b, Op::ADD, ctx);
}

tensor tensor::sub(const tensor& a, const tensor& b, core::context& ctx) {
    return tensor::bop(a, b, Op::SUB, ctx);
}

tensor tensor::mul(const tensor& a, const tensor& b, core::context& ctx) {
    return tensor::bop(a, b, Op::MUL, ctx);
}

tensor tensor::div(const tensor& a, const tensor& b, core::context& ctx) {
    return tensor::bop(a, b, Op::DIV, ctx);
}

// ----- Matmul -----
tensor tensor::matmul(const tensor& a, const tensor& b, core::context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    COMTAM_CHECK_AND_THROW(a.dtype_ == b.dtype_, std::runtime_error,
                           "Two operands must have the same dtype");

    COMTAM_CHECK_AND_THROW(a.shape().size() == 2 && b.shape().size() == 2, std::runtime_error,
                           "Right now, matmul only supports 2D array");

    COMTAM_CHECK_AND_THROW(a.shape()[1] == b.shape()[0], std::runtime_error,
                           "To do matmul, column of a must match with row of b");

    COMTAM_LOG_DEBUG("matmul {}_{}:\na={}\nb={}\n", core::op2kernel(Op::MATMUL),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

    tensor out({a.shape()[0], b.shape()[1]}, device, a.dtype_);

    core::command_desc cmd = {
        .kernel = {.op = Op::MATMUL, .dtype = a.dtype_},
        .a = {.storage = a.storage_.get(), .view = core::view_desc::from_view(a.view_)},
        .b = {.storage = b.storage_.get(), .view = core::view_desc::from_view(b.view_)},
        .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("matmul ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    device.submit_matmul(cmd, kernels);

    return out;
}

// ----- View operation -----
// These operations are free (change view only)

tensor tensor::permute(const view_vector& new_axes) const {
    return tensor(storage_, view_.permute(new_axes), dtype_);
}

tensor tensor::transpose(view_int a, view_int b) const {
    return tensor(storage_, view_.transpose(a, b), dtype_);
}

tensor tensor::shrink(const pair_view_vector& limits) const {
    return tensor(storage_, view_.shrink(limits), dtype_);
}

tensor tensor::expand(const view_vector& new_shape) const {
    return tensor(storage_, view_.expand(new_shape), dtype_);
}

tensor tensor::reshape(const view_vector& new_shape) const {
    return tensor(storage_, view_.reshape(new_shape), dtype_);
}
