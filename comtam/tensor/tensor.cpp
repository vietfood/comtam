/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include "comtam/tensor/tensor.h"

#include "comtam/macros/log.h"
#include "comtam/tensor/impl.h"
#include "comtam/utils/debug.h"

using namespace comtam;

tensor tensor::bop(const tensor& a, const tensor& b, const Op& op) {
    utils::check_same_runtime(a.impl_->runtime_state, b.impl_->runtime_state);
    auto runtime_state = resolve_runtime(a.impl_.get(), b.impl_.get());

    auto& device = runtime_state->device();
    auto& kernels = runtime_state->kernels();

    auto final_shape =
        utils::check_binary(a.impl_->view, a.impl_->dtype, b.impl_->view, b.impl_->dtype);

    COMTAM_LOG_DEBUG("binop {}_{}:\na={}\nb={}\n", core::op2kernel(op),
                     core::dtype2kernel(a.impl_->dtype), utils::format_view(a.impl_->view),
                     utils::format_view(b.impl_->view));

    // expand view of a and b based on broadcast shape
    auto a_expand = a.impl_->view.expand(final_shape);
    auto b_expand = b.impl_->view.expand(final_shape);

    COMTAM_LOG_DEBUG("binop broadcast: shape={}\na_expand={}\nb_expand={}\n",
                     utils::format_view_vector(final_shape), utils::format_view(a_expand),
                     utils::format_view(b_expand));

    auto out_impl = make_impl_from_view(final_shape, a.impl_->dtype, runtime_state);

    core::command_desc cmd = {
        .kernel = {.op = op, .dtype = a.impl_->dtype},
        .a = {.storage = a.impl_->storage.get(), .view = core::view_desc::from_view(a_expand)},
        .b = {.storage = b.impl_->storage.get(), .view = core::view_desc::from_view(b_expand)},
        .out_buffer = out_impl->storage.get()};

    COMTAM_LOG_DEBUG("bop ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out_impl->storage->size());

    device.submit_bop(cmd, kernels);

    return tensor(std::move(out_impl));
}

tensor tensor::uop(const tensor& a, const Op& op) {
    auto runtime_state = a.impl_->runtime_state;

    auto& device = runtime_state->device();
    auto& kernels = runtime_state->kernels();

    utils::check_unary(a.impl_->view, a.impl_->dtype);

    COMTAM_LOG_DEBUG("uop {}_{}:\na={}\n", core::op2kernel(op), core::dtype2kernel(a.impl_->dtype),
                     utils::format_view(a.impl_->view));

    auto out_impl = make_impl_from_view(a.impl_->view.shape, a.impl_->dtype, runtime_state);

    core::command_desc cmd = {
        .kernel = {.op = op, .dtype = a.impl_->dtype},
        .a = {.storage = a.impl_->storage.get(), .view = core::view_desc::from_view(a.impl_->view)},
        .b = {},
        .out_buffer = out_impl->storage.get()};

    COMTAM_LOG_DEBUG("uop ViewInfo:\na: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), out_impl->storage->size());

    device.submit_uop(cmd, kernels);

    return tensor(std::move(out_impl));
}

tensor tensor::rop(const tensor& a, const Op& op, view_int dim, bool keep_dim,
                   const OpVariant& variant) {
    auto runtime_state = a.impl_->runtime_state;
    auto& device = runtime_state->device();
    auto& kernels = runtime_state->kernels();

    view_vector final_shape;
    if (variant == OpVariant::AXIS) {
        final_shape = utils::check_reduce_axis(a.impl_->view, a.impl_->dtype, dim, keep_dim);
    } else {
        final_shape = utils::check_reduce_full(a.impl_->view, a.impl_->dtype);
    }

    COMTAM_LOG_DEBUG("rop {}_{}:\na={}\n", core::op2kernel(op), core::dtype2kernel(a.impl_->dtype),
                     utils::format_view(a.impl_->view));

    auto out_impl = make_impl_from_view(final_shape, a.impl_->dtype, runtime_state);

    core::command_desc cmd = {
        .kernel = {.op = op, .dtype = a.impl_->dtype, .variant = variant},
        .a = {.storage = a.impl_->storage.get(), .view = core::view_desc::from_view(a.impl_->view)},
        .b = {},
        .out_buffer = out_impl->storage.get(),
        .extra = {.axis = variant == OpVariant::AXIS ? static_cast<int64_t>(dim) : -1}};

    COMTAM_LOG_DEBUG("rop ViewInfo:\na: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), out_impl->storage->size());

    device.submit_reduce(cmd, kernels);

    return tensor(std::move(out_impl));
}

tensor tensor::matmul(const tensor& a, const tensor& b) {
    utils::check_same_runtime(a.impl_->runtime_state, b.impl_->runtime_state);
    auto runtime_state = resolve_runtime(a.impl_.get(), b.impl_.get());

    auto& device = runtime_state->device();
    auto& kernels = runtime_state->kernels();

    auto out_shape =
        utils::check_matmul(a.impl_->view, a.impl_->dtype, b.impl_->view, b.impl_->dtype);

    COMTAM_LOG_DEBUG("matmul {}_{}:\na={}\nb={}\n", core::op2kernel(Op::MATMUL),
                     core::dtype2kernel(a.impl_->dtype), utils::format_view(a.impl_->view),
                     utils::format_view(b.impl_->view));

    bool is_contiguous = (a.impl_->view.is_contiguous() && a.impl_->view.offset == 0) &&
                         (b.impl_->view.is_contiguous() && b.impl_->view.offset == 0);

    OpVariant variant = is_contiguous ? OpVariant::CONTIGUOUS : OpVariant::STRIDED;

    auto out_impl = make_impl_from_view(out_shape, a.impl_->dtype, runtime_state);

    core::command_desc cmd = {
        .kernel = {.op = Op::MATMUL, .dtype = a.impl_->dtype, .variant = variant},
        .a = {.storage = a.impl_->storage.get(), .view = core::view_desc::from_view(a.impl_->view)},
        .b = {.storage = b.impl_->storage.get(), .view = core::view_desc::from_view(b.impl_->view)},
        .out_buffer = out_impl->storage.get()};

    COMTAM_LOG_DEBUG("matmul ViewInfo:\na: {}\nb: {}\nout_bytes={} (output written linearly)\n",
                     utils::format_view_info(cmd.a.view), utils::format_view_info(cmd.b.view),
                     out_impl->storage->size());

    device.submit_matmul(cmd, kernels);

    return tensor(std::move(out_impl));
}