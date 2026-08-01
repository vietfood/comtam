/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include "comtam/tensor/tensor.h"

#include "comtam/core/command.h"
#include "comtam/core/context.h"
#include "comtam/macros/log.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/op.h"
#include "comtam/utils/debug.h"

using namespace comtam;

// ----- Binary operation -----
tensor tensor::bop(const tensor& a, const tensor& b, const Op& op, core::context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    auto final_shape = checks::check_binary(a.view_, a.dtype_, b.view_, b.dtype_);

    COMTAM_LOG_DEBUG("binop {}_{}:\na={}\nb={}\n", core::op2kernel(op),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

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

    device.submit_bop(cmd, kernels);

    COMTAM_LOG_DEBUG("binop ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    return out;
}

// ----- Unary operation ---

tensor tensor::uop(const tensor& a, const Op& op, core::context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    checks::check_unary(a.view_, a.dtype_);

    COMTAM_LOG_DEBUG("uop {}_{}:\na={}\n", core::op2kernel(op), core::dtype2kernel(a.dtype_),
                     utils::format_view(a.view_));

    tensor out(a.view_.shape, device, a.dtype_);

    core::command_desc cmd = {
        .kernel = {.op = op, .dtype = a.dtype_},
        .a = {.storage = a.storage_.get(), .view = core::view_desc::from_view(a.view_)},
        .b = {},  // default
        .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("binop ViewInfo:\na: {}\n\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), out.storage_->size());

    device.submit_uop(cmd, kernels);

    return out;
}

// ----- Matmul -----
tensor tensor::matmul(const tensor& a, const tensor& b, core::context& ctx) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    auto out_shape = checks::check_matmul(a.view_, a.dtype_, b.view_, b.dtype_);

    COMTAM_LOG_DEBUG("matmul {}_{}:\na={}\nb={}\n", core::op2kernel(Op::MATMUL),
                     core::dtype2kernel(a.dtype_), utils::format_view(a.view_),
                     utils::format_view(b.view_));

    tensor out(out_shape, device, a.dtype_);

    // check contiguous
    bool is_contiguous = (a.view_.is_contiguous() && a.offset() == 0) &&
                         (b.view_.is_contiguous() && b.offset() == 0);
    OpVariant variant = is_contiguous ? OpVariant::CONTIGUOUS : OpVariant::STRIDED;

    core::command_desc cmd = {
        .kernel = {.op = Op::MATMUL, .dtype = a.dtype_, .variant = variant},
        .a = {.storage = a.storage_.get(), .view = core::view_desc::from_view(a.view_)},
        .b = {.storage = b.storage_.get(), .view = core::view_desc::from_view(b.view_)},
        .out_buffer = out.storage_.get()};

    COMTAM_LOG_DEBUG("matmul ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out.storage_->size());

    device.submit_matmul(cmd, kernels);

    return out;
}

// ---- Reduction operation -----
tensor tensor::rop(const tensor& a, const Op& op, core::context& ctx, view_int dim, bool keep_dim,
                   const OpVariant& variant) {
    auto& device = ctx.device();
    auto& kernels = ctx.kernels();

    view_vector final_shape;
    if (variant == OpVariant::AXIS) {
        final_shape = checks::check_reduce_axis(a.view_, a.dtype_, dim, keep_dim);
    } else {
        final_shape = checks::check_reduce_full(a.view_, a.dtype_);
    }

    COMTAM_LOG_DEBUG("reduce_op {}_{}:\na={}\n", core::op2kernel(op), core::dtype2kernel(a.dtype_),
                     utils::format_view(a.view_));

    tensor out(final_shape, device, a.dtype_);

    core::command_desc cmd = {
        .kernel = {.op = op, .dtype = a.dtype_, .variant = variant},
        .a = {.storage = a.storage_.get(), .view = core::view_desc::from_view(a.view_)},
        .b = {},
        .out_buffer = out.storage_.get(),
        .extra = {.axis = variant == OpVariant::AXIS ? static_cast<int64_t>(dim) : -1}};

    COMTAM_LOG_DEBUG("reduce_op ViewInfo:\na: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), out.storage_->size());

    device.submit_reduce(cmd, kernels);

    return out;
}
