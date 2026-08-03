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

#pragma once

#include <memory>

#include "comtam/core/context.h"
#include "comtam/tensor/impl.h"
#include "comtam/utils/checks.h"

namespace comtam {

#ifdef COMTAM_BUILD_TESTS
namespace tests {
struct tensor_access;
}
#endif

class tensor {
   public:
    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;
    ~tensor() = default;

    bool is_same(const tensor& other) const noexcept { return impl_ == other.impl_; }

#ifdef COMTAM_BUILD_TESTS
    friend struct tests::tensor_access;
#endif

    /**
     * Initialize Tensor from an array
     */
    template <typename T>
    explicit tensor(const T* data, const view_vector& shape, DType dtype = DType::Float32,
                    const core::context& context = core::default_context())
        : impl_(nullptr) {
        impl_ = make_impl_from_array(data, shape, dtype, context.state());
    }

    /**
     * Initialize Tensor from a scalar
     */
    template <typename T>
    explicit tensor(T value, DType dtype = DType::Float32,
                    const core::context& context = core::default_context())
        : impl_(nullptr) {
        impl_ = make_impl_from_scalar(value, dtype, context.state());
    }

    /**
     * Initialize Tensor from a view
     */
    explicit tensor(const view& view, DType dtype = DType::Float32,
                    const core::context& context = core::default_context())
        : impl_(nullptr) {
        impl_ = make_impl_from_view(view, dtype, context.state());
    }

    /**
     * Initialize Tensor from a shape
     */
    explicit tensor(const view_vector& shape, DType dtype = DType::Float32,
                    const core::context& context = core::default_context())
        : impl_(nullptr) {
        impl_ = make_impl_from_shape(shape, dtype, context.state());
    }

    /**
     * Get data from a vector for an initialized Tensor
     */
    template <typename T>
    void from_vector(const std::vector<T>& data) {
        COMTAM_DISPATCH_DTYPE(impl_->dtype, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "dtype of vector didn't match with input Tensor");
            }

            auto numel = static_cast<size_int>(impl_->view.numel());
            COMTAM_CHECK_AND_THROW(data.size() == numel, std::runtime_error,
                                   "Tensor::from_vector: data size does not match tensor size");

            COMTAM_CHECK_AND_THROW(impl_->view.is_contiguous(), std::runtime_error,
                                   "Tensor::from_vector: non-contiguous tensor writes are not "
                                   "supported");

            auto device = impl_->runtime_state->device();
            device.copy<scalar_t>(data.data(), numel, *impl_->storage.get());
        });
    }

    void from_vector(const std::vector<float>& data) { from_vector<float>(data); }

    /**
     * Return a vector from a Tensor
     */
    template <typename T>
    std::vector<T> to_vector() const {
        return COMTAM_DISPATCH_DTYPE(impl_->dtype, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "dtype of vector didn't match with input Tensor");
            }
            COMTAM_CHECK_AND_THROW(impl_, std::runtime_error,
                                   "Tensor::to_vector: implementation is not initialized");

            std::vector<scalar_t> result(static_cast<size_int>(impl_->view.numel()));

            for (size_int i = 0; i < result.size(); ++i) {
                result[i] = impl_->storage->at<scalar_t>(
                    impl_->view.physical_offset(static_cast<view_int>(i)));
            }

            return result;
        });
    }

    std::vector<float> to_vector() const { return to_vector<float>(); }

    // ----- Some common getters -----

    /**
     * Read one logical element by linear index.
     */
    template <typename T>
    T at(view_int i) const {
        return COMTAM_DISPATCH_DTYPE(impl_->dtype, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "dtype of vector didn't match with input Tensor");
            }
            COMTAM_CHECK_AND_THROW(impl_, std::runtime_error,
                                   "Tensor::at: implementation is not initialized");
            return impl_->storage->at<scalar_t>(impl_->view.physical_offset(i));
        });
    }

    size_int numel() const { return static_cast<size_int>(impl_->view.numel()); }
    size_int dim() const { return static_cast<size_int>(impl_->view.dim()); }
    size_int offset() const { return static_cast<size_int>(impl_->view.offset); }
    view_vector shape() const { return impl_->view.shape; }
    view_vector strides() const { return impl_->view.strides; }
    DType dtype() const { return impl_->dtype; }

    // ----- View operation -----

    tensor permute(const view_vector& new_axes) const {
        auto new_view = impl_->view.permute(new_axes);
        auto new_impl =
            make_impl_from_storage(impl_->storage, new_view, impl_->dtype, impl_->runtime_state);
        return tensor(std::move(new_impl));
    }

    tensor transpose(view_int a, view_int b) const {
        auto new_view = impl_->view.transpose(a, b);
        auto new_impl =
            make_impl_from_storage(impl_->storage, new_view, impl_->dtype, impl_->runtime_state);
        return tensor(std::move(new_impl));
    }

    tensor shrink(const pair_view_vector& limits) const {
        auto new_view = impl_->view.shrink(limits);
        auto new_impl =
            make_impl_from_storage(impl_->storage, new_view, impl_->dtype, impl_->runtime_state);
        return tensor(std::move(new_impl));
    }

    tensor expand(const view_vector& new_shape) const {
        auto new_view = impl_->view.expand(new_shape);
        auto new_impl =
            make_impl_from_storage(impl_->storage, new_view, impl_->dtype, impl_->runtime_state);
        return tensor(std::move(new_impl));
    }

    tensor reshape(const view_vector& new_shape) const {
        auto new_view = impl_->view.reshape(new_shape);
        auto new_impl =
            make_impl_from_storage(impl_->storage, new_view, impl_->dtype, impl_->runtime_state);
        return tensor(std::move(new_impl));
    }

    // ----- Binary operation -----

    static tensor add(const tensor& a, const tensor& b) { return tensor::bop(a, b, Op::ADD); }

    static tensor sub(const tensor& a, const tensor& b) {
        utils::check_same_runtime(a.impl_->runtime_state, b.impl_->runtime_state);
        utils::check_binary(a.impl_->view, a.impl_->dtype, b.impl_->view, b.impl_->dtype);
        return add(a, tensor::neg(b));
    }

    static tensor mul(const tensor& a, const tensor& b) { return tensor::bop(a, b, Op::MUL); }

    static tensor div(const tensor& a, const tensor& b) {
        utils::check_same_runtime(a.impl_->runtime_state, b.impl_->runtime_state);
        utils::check_binary(a.impl_->view, a.impl_->dtype, b.impl_->view, b.impl_->dtype);
        return mul(a, tensor::recip(b));
    }

    // ----- Unary operation -----

    static tensor neg(const tensor& a) { return tensor::uop(a, Op::NEG); }

    static tensor recip(const tensor& a) { return tensor::uop(a, Op::RECIP); }

    // ----- Axis Reduction -----

    static tensor sum(const tensor& a, view_int dim, bool keep_dim = false) {
        return tensor::rop(a, Op::SUM, dim, keep_dim, OpVariant::AXIS);
    }

    static tensor max(const tensor& a, view_int dim, bool keep_dim = false) {
        return tensor::rop(a, Op::MAX, dim, keep_dim, OpVariant::AXIS);
    }

    static tensor min(const tensor& a, view_int dim, bool keep_dim = false) {
        utils::check_reduce_axis(a.impl_->view, a.impl_->dtype, dim, keep_dim);
        return neg(max(neg(a), dim, keep_dim));
    }

    static tensor mean(const tensor& a, view_int dim, bool keep_dim = false) {
        return COMTAM_DISPATCH_DTYPE(a.impl_->dtype, [&] {
            utils::check_reduce_axis(a.impl_->view, a.impl_->dtype, dim, keep_dim);
            auto scale_impl = make_impl_from_scalar(
                static_cast<scalar_t>(1.0f / a.impl_->view.shape[static_cast<size_int>(dim)]),
                a.impl_->dtype, a.impl_->runtime_state);
            return mul(sum(a, dim, keep_dim), tensor(std::move(scale_impl)));
        });
    }

    // ----- Full Reduction -----

    static tensor sum(const tensor& a) {
        return tensor::rop(a, Op::SUM, -1, false, OpVariant::FULL);
    }

    static tensor max(const tensor& a) {
        return tensor::rop(a, Op::MAX, -1, false, OpVariant::FULL);
    }

    static tensor min(const tensor& a) { return neg(max(neg(a))); }

    static tensor mean(const tensor& a) {
        return COMTAM_DISPATCH_DTYPE(a.impl_->dtype, [&] {
            utils::check_reduce_full(a.impl_->view, a.impl_->dtype);
            auto scale_impl =
                make_impl_from_scalar(static_cast<scalar_t>(1.0f / a.impl_->view.numel()),
                                      a.impl_->dtype, a.impl_->runtime_state);
            return mul(sum(a), tensor(std::move(scale_impl)));
        });
    }

    // ----- Matmul -----
    static tensor matmul(const tensor& a, const tensor& b);

   private:
    // ----- Private methods -----
    explicit tensor(std::shared_ptr<tensor_impl> impl) : impl_(std::move(impl)) {
        COMTAM_CHECK_AND_THROW(impl_, std::runtime_error, "tensor requires an implementation");
    }

    static tensor bop(const tensor& a, const tensor& b, const Op& op);
    static tensor uop(const tensor& a, const Op& op);
    static tensor rop(const tensor& a, const Op& op, view_int dim = -1, bool keep_dim = false,
                      const OpVariant& variant = OpVariant::FULL);

    // ----- Private variables -----

    std::shared_ptr<tensor_impl> impl_;
};

}  // namespace comtam