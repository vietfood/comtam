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

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/tensor/checks.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/op.h"
#include "comtam/tensor/view.h"
#include "comtam/types.h"

namespace comtam {
class tensor {
   public:
    /**
     * Initialize Tensor from an array
     */
    template <typename T>
    tensor(const T* data, const view_vector& shape, core::metal_device& device,
           DType dtype = DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error, "DType didn't match with input data");
            }

            // create a storage from device first
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_int>(view_.numel()));
            device.copy<scalar_t>(data, static_cast<size_int>(view_.numel()), buffer);

            // then wrap to shared_ptr
            storage_ = std::make_shared<core::storage>(std::move(buffer));
        });
    }

    /**
     * Initialize from a scalar
     */
    template <typename T>
    tensor(T value, core::metal_device& device, DType dtype = DType::Float32)
        : dtype_(dtype), view_({}), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error, "DType didn't match with input data");
            }

            auto buffer = device.allocate(sizeof(scalar_t));
            device.copy<scalar_t>(value, buffer);
            storage_ = std::make_shared<core::storage>(std::move(buffer));
        });
    }

    /**
     * Initialize an empty Tensor
     */

    // init from a shape (always contiguous and offset = 0)
    tensor(const view_vector& shape, core::metal_device& device, DType dtype = DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            // we create an empty buffer
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_int>(view_.numel()));
            storage_ = std::make_shared<core::storage>(std::move(buffer));
        });
    }

    tensor(const view& view, core::metal_device& device, DType dtype = DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_int>(view_.numel()));
            storage_ = std::make_shared<core::storage>(std::move(buffer));
        });
    }

    /**
     * Initialize from a Storage
     * Note: we pass by const reference so `storage_ = storage` will trigger a copy
     * => storage_ and storage both point to underlying object (new reference)
     */
    tensor(const std::shared_ptr<core::storage>& storage, const view_vector& shape,
           DType dtype = DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            // we must ensure the storage byte match the view
            COMTAM_CHECK_AND_THROW(
                sizeof(scalar_t) * static_cast<size_int>(view_.numel()) == storage->size(),
                std::runtime_error, "Storage size doesn't match dtype and shape");
            // then we only need to set target
            storage_ = storage;
        });
    }

    tensor(const std::shared_ptr<core::storage>& storage, const view& view,
           DType dtype = DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        /*
         * A view can have a different logical numel from its aliased storage.
         * Readback uses View::physical_offset to map logical indices to storage.
         */
        storage_ = storage;
    }

    /**
     * Get data from a vector for an initialized Tensor
     */
    template <typename T>
    void from_vector(const std::vector<T>& data, core::metal_device& device) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "dtype of vector didn't match with input Tensor");
            }
            auto numel = static_cast<size_int>(view_.numel());
            COMTAM_CHECK_AND_THROW(data.size() == numel, std::runtime_error,
                                   "Tensor::from_vector: data size does not match tensor size");

            COMTAM_CHECK_AND_THROW(view_.is_contiguous(), std::runtime_error,
                                   "Tensor::from_vector: non-contiguous tensor writes are not "
                                   "supported");

            device.copy<scalar_t>(data.data(), data.size(), *storage_.get());
        });
    }

    void from_vector(const std::vector<float>& data, core::metal_device& device) {
        from_vector<float>(data, device);
    }

    /**
     * Return a vector from a Tensor
     */
    template <typename T>
    std::vector<T> to_vector(core::metal_device&) const {
        return COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "dtype of vector didn't match with input Tensor");
            }
            COMTAM_CHECK_AND_THROW(storage_, std::runtime_error,
                                   "Tensor::to_vector: storage is not allocated");

            std::vector<scalar_t> result(static_cast<size_int>(view_.numel()));
            // instead of copy, we must gather the data for "the correct shape"
            for (size_int i = 0; i < result.size(); ++i) {
                result[i] = storage_->at<scalar_t>(view_.physical_offset(static_cast<view_int>(i)));
            }
            return result;
        });
    }

    std::vector<float> to_vector(core::metal_device& device) const {
        return to_vector<float>(device);
    }

    // ----- Some common getters -----

    /**
     * Read one logical element by linear index.
     * Uses View::physical_offset so non-contiguous views stay correct.
     */
    template <typename T>
    T at(view_int i) const {
        return COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error,
                                   "Tensor::at: dtype did not match requested type");
            }
            COMTAM_CHECK_AND_THROW(storage_, std::runtime_error,
                                   "Tensor::at: storage is not allocated");
            return storage_->at<scalar_t>(view_.physical_offset(i));
        });
    }

    size_int numel() const { return static_cast<size_int>(view_.numel()); }
    size_int dim() const { return static_cast<size_int>(view_.dim()); }
    size_int offset() const { return static_cast<size_int>(view_.offset); }
    view_vector shape() const { return view_.shape; }
    view_vector strides() const { return view_.strides; }
    DType dtype() const { return dtype_; }

    // ----- View operation -----

    tensor permute(const view_vector& new_axes) const {
        return tensor(storage_, view_.permute(new_axes), dtype_);
    }

    tensor transpose(view_int a, view_int b) const {
        return tensor(storage_, view_.transpose(a, b), dtype_);
    }

    tensor shrink(const pair_view_vector& limits) const {
        return tensor(storage_, view_.shrink(limits), dtype_);
    }

    tensor expand(const view_vector& new_shape) const {
        return tensor(storage_, view_.expand(new_shape), dtype_);
    }

    tensor reshape(const view_vector& new_shape) const {
        return tensor(storage_, view_.reshape(new_shape), dtype_);
    }

    // ----- Binary operation -----

    static tensor add(const tensor& a, const tensor& b, core::context& ctx) {
        return tensor::bop(a, b, Op::ADD, ctx);
    }

    static tensor sub(const tensor& a, const tensor& b, core::context& ctx) {
        checks::check_binary(a.view_, a.dtype_, b.view_, b.dtype_);
        return add(a, tensor::neg(b, ctx), ctx);
    }

    static tensor mul(const tensor& a, const tensor& b, core::context& ctx) {
        return tensor::bop(a, b, Op::MUL, ctx);
    }

    static tensor div(const tensor& a, const tensor& b, core::context& ctx) {
        checks::check_binary(a.view_, a.dtype_, b.view_, b.dtype_);
        return tensor::mul(a, tensor::recip(b, ctx), ctx);
    }

    // ----- Unary operation -----

    static tensor neg(const tensor& a, core::context& ctx) { return tensor::uop(a, Op::NEG, ctx); }

    static tensor recip(const tensor& a, core::context& ctx) {
        return tensor::uop(a, Op::RECIP, ctx);
    }

    // ----- Matmul -----

    static tensor matmul(const tensor& a, const tensor& b, core::context& ctx);

    // ----- Axis Reduction -----
    // Current code can reduce along one dimension only.

    static tensor sum(const tensor& a, view_int dim, bool keep_dim, core::context& ctx) {
        return rop(a, Op::SUM, ctx, dim, keep_dim, OpVariant::AXIS);
    }

    static tensor max(const tensor& a, view_int dim, bool keep_dim, core::context& ctx) {
        return rop(a, Op::MAX, ctx, dim, keep_dim, OpVariant::AXIS);
    }

    static tensor mean(const tensor& a, view_int dim, bool keep_dim, core::context& ctx) {
        return COMTAM_DISPATCH_DTYPE(a.dtype_, [&] {
            checks::check_reduce_axis(a.view_, a.dtype_, dim, keep_dim);
            auto scale = tensor(static_cast<scalar_t>(a.view_.shape[static_cast<size_int>(dim)]), ctx.device(), a.dtype());
            return tensor::div(tensor::sum(a, dim, keep_dim, ctx),
                               scale,
                               ctx);
        });
    }

    static tensor min(const tensor& a, view_int dim, bool keep_dim, core::context& ctx) {
        // min({a, b, ...}) = -max({-a, -b, ...})
        checks::check_reduce_axis(a.view_, a.dtype_, dim, keep_dim);
        return neg(max(neg(a, ctx), dim, keep_dim, ctx), ctx);
    }

    // ---- Full Reduction -----

    static tensor sum(const tensor& a, core::context& ctx) {
        return rop(a, Op::SUM, ctx, -1, false, OpVariant::FULL);
    }

    static tensor mean(const tensor& a, core::context& ctx) {
        return COMTAM_DISPATCH_DTYPE(a.dtype_, [&] {
            checks::check_reduce_full(a.view_, a.dtype_);
            auto scale = tensor(static_cast<scalar_t>(a.numel()), ctx.device(), a.dtype());
            return tensor::div(tensor::sum(a, ctx), scale, ctx);
        });
    }

    static tensor max(const tensor& a, core::context& ctx) {
        return rop(a, Op::MAX, ctx, -1, false, OpVariant::FULL);
    }

    static tensor min(const tensor& a, core::context& ctx) {
        checks::check_reduce_full(a.view_, a.dtype_);
        return neg(max(neg(a, ctx), ctx), ctx);
    }

   private:
    // --- Private methods ---
    static tensor bop(const tensor& a, const tensor& b, const Op& op, core::context& ctx);
    static tensor uop(const tensor& a, const Op& op, core::context& ctx);
    static tensor rop(const tensor& a, const Op& op, core::context& ctx, view_int dim = -1,
                      bool keep_dim = false, const OpVariant& variant = OpVariant::FULL);

    // --- Private variables ---
    DType dtype_;
    view view_;
    std::shared_ptr<core::storage> storage_;
};
}  // namespace comtam
