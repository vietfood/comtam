#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "comtam/core/context.h"
#include "comtam/core/dtype.h"
#include "comtam/core/ops.h"
#include "comtam/core/storage.h"
#include "comtam/core/view.h"
#include "comtam/macros/log.h"

namespace comtam {
class Tensor {
   public:
    /**
     * Initialize Tensor from an array
     */
    template <typename T>
    Tensor(const T* data, std::vector<core::ViewInt> shape, core::Device& device,
           core::DType dtype = core::DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            if constexpr (!std::is_same_v<T, scalar_t>) {
                COMTAM_THROW_ERROR(std::runtime_error, "DType didn't match with input data");
            }

            // create a storage from device first
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
            device.copy<scalar_t>(data, static_cast<size_t>(view_.numel()), buffer);

            // then wrap to shared_ptr
            storage_ = std::make_shared<core::Storage>(std::move(buffer));
        });
    }

    /**
     * Initialize an empty Tensor
     */

    // init from a shape (always contiguous and offset = 0)
    Tensor(const std::vector<core::ViewInt>& shape, core::Device& device,
           core::DType dtype = core::DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            // we create an empty buffer
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
            storage_ = std::make_shared<core::Storage>(std::move(buffer));
        });
    }

    Tensor(const core::View& view, core::Device& device, core::DType dtype = core::DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_t>(view_.numel()));
            storage_ = std::make_shared<core::Storage>(std::move(buffer));
        });
    }

    /**
     * Initialize from a Storage
     * Note: we pass by const reference so `storage_ = storage` will trigger a copy
     * => storage_ and storage both point to underlying object (new reference)
     */
    Tensor(const std::shared_ptr<core::Storage>& storage, const std::vector<core::ViewInt>& shape,
           core::DType dtype = core::DType::Float32)
        : dtype_(dtype), view_(shape), storage_(nullptr) {
        COMTAM_DISPATCH_DTYPE(dtype_, [&] {
            // we must ensure the storage byte match the Tensor
            COMTAM_CHECK_AND_THROW(sizeof(scalar_t) * static_cast<size_t>(view_.numel()) ==
                                       storage->size(),
                                   std::runtime_error,
                                   "Storage size doesn't match dtype and shape");
            // then we only need to set target
            storage_ = storage;
        });
    }

    Tensor(const std::shared_ptr<core::Storage>& storage, const core::View& view,
           core::DType dtype = core::DType::Float32)
        : dtype_(dtype), view_(view), storage_(nullptr) {
        // A view can have a different logical numel from its aliased storage.
        // Readback uses View::physical_offset to map logical indices to storage.
        storage_ = storage;
    }

    /**
     * Get data from a vector for an initialized Tensor
     */
    template <typename T>
    void from_vector(const std::vector<T>& data, core::Device& device) {
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

    void from_vector(const std::vector<float>& data, core::Device& device) {
        from_vector<float>(data, device);
    }

    /**
     * Return a vector from a Tensor
     */
    template <typename T>
    std::vector<T> to_vector(core::Device&) const {
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

    std::vector<float> to_vector(core::Device& device) const { return to_vector<float>(device); }

    size_t numel() const { return static_cast<size_t>(view_.numel()); }
    size_t dim() const { return view_.dim(); }
    std::vector<core::ViewInt> shape() const { return view_.shape; }
    std::vector<core::ViewInt> strides() const { return view_.strides; }
    core::DType dtype() const { return dtype_; }

    // ----- View operation -----
    Tensor permute(const std::vector<int64_t>& new_axis) const;
    Tensor transpose(int64_t a, int64_t b) const;
    Tensor shrink(const std::vector<std::pair<int64_t, int64_t>>& limits) const;
    Tensor expand(const std::vector<int64_t>& new_shape) const;
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    // ----- Binary operation -----
    static Tensor bop(const Tensor& a, const Tensor& b, const core::Op& op, core::Context& ctx);
    static Tensor add(const Tensor& a, const Tensor& b, core::Context& ctx);
    static Tensor sub(const Tensor& a, const Tensor& b, core::Context& ctx);
    static Tensor mul(const Tensor& a, const Tensor& b, core::Context& ctx);
    static Tensor div(const Tensor& a, const Tensor& b, core::Context& ctx);

    // ----- Other operation -----
    static Tensor matmul(const Tensor& a, const Tensor& b, core::Context& ctx);

   private:
    core::DType dtype_;
    core::View view_;
    std::shared_ptr<core::Storage> storage_;
};
}  // namespace comtam
