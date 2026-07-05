#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/op.h"
#include "comtam/tensor/view.h"

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
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
            device.copy<scalar_t>(data, static_cast<size_t>(view_.numel()), buffer);

            // then wrap to shared_ptr
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
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
            storage_ = std::make_shared<core::storage>(std::move(buffer));
        });
    }

    tensor(const view& view, core::metal_device& device, DType dtype = DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
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
            // we must ensure the storage byte match the Tensor
            COMTAM_CHECK_AND_THROW(
                sizeof(scalar_t) * static_cast<size_t>(view_.numel()) == storage->size(),
                std::runtime_error, "Storage size doesn't match dtype and shape");
            // then we only need to set target
            storage_ = storage;
        });
    }

    tensor(const std::shared_ptr<core::storage>& storage, const view& view,
           DType dtype = DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        // A view can have a different logical numel from its aliased storage.
        // Readback uses View::physical_offset to map logical indices to storage.
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

            COMTAM_CHECK_AND_THROW(storage_, std::runtime_error,
                                   "Tensor::from_vector: storage is not allocated");

            auto numel = static_cast<size_t>(view_.numel());
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

            std::vector<scalar_t> result(static_cast<size_t>(view_.numel()));
            // instead of copy, we must gather the data for "the correct shape"
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = storage_->at<scalar_t>(view_.physical_offset(i));
            }
            return result;
        });
    }

    std::vector<float> to_vector(core::metal_device& device) const {
        return to_vector<float>(device);
    }

    size_t numel() const { return static_cast<size_t>(view_.numel()); }
    size_t dim() const { return view_.dim(); }
    view_vector shape() const { return view_.shape; }
    view_vector strides() const { return view_.strides; }
    DType dtype() const { return dtype_; }

    // ----- View operation -----
    tensor permute(const view_vector& new_axes) const;
    tensor transpose(view_int a, view_int b) const;
    tensor shrink(const pair_view_vector& limits) const;
    tensor expand(const view_vector& new_shape) const;
    tensor reshape(const view_vector& new_shape) const;

    // ----- Binary operation -----
    static tensor add(const tensor& a, const tensor& b, core::context& ctx);
    static tensor sub(const tensor& a, const tensor& b, core::context& ctx);
    static tensor mul(const tensor& a, const tensor& b, core::context& ctx);
    static tensor div(const tensor& a, const tensor& b, core::context& ctx);

    // ----- Other operation -----
    static tensor matmul(const tensor& a, const tensor& b, core::context& ctx);

   private:
    // --- Private methods ---
    static tensor bop(const tensor& a, const tensor& b, const Op& op, core::context& ctx);

    // --- Private variables ---
    DType dtype_;
    view view_;
    std::shared_ptr<core::storage> storage_;
};
}  // namespace comtam
