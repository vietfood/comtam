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
#include <stdexcept>
#include <utility>

#include "comtam/core/context.h"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/view.h"

namespace comtam {
struct tensor_impl {
    explicit tensor_impl(DType dtype_value, const view& view_value,
                         std::shared_ptr<core::storage> storage_value,
                         std::shared_ptr<core::runtime_state> runtime_state_value)
        : runtime_state(std::move(runtime_state_value)),
          storage(std::move(storage_value)),
          dtype(dtype_value),
          view(view_value) {
        COMTAM_CHECK_AND_THROW(runtime_state, std::runtime_error,
                               "tensor_impl requires runtime_state");
        COMTAM_CHECK_AND_THROW(storage, std::runtime_error, "tensor_impl requires storage");
    }

    // we declare based on destruction order
    std::shared_ptr<core::runtime_state> runtime_state;
    std::shared_ptr<core::storage> storage;
    DType dtype;
    view view;
};

// raw tensor_impl == less headache
static std::shared_ptr<core::runtime_state> resolve_runtime(tensor_impl* lhs, tensor_impl* rhs) {
    // right now, we trust the caller that the tensors have the same runtime
    (void)rhs;
    return lhs->runtime_state;
}

template <typename T>
static std::shared_ptr<tensor_impl> make_impl_from_array(
    const T* data, const view_vector& shape, DType dtype,
    std::shared_ptr<core::runtime_state> runtime_state) {
    return COMTAM_DISPATCH_DTYPE(dtype, [&] {
        if constexpr (!std::is_same_v<T, scalar_t>) {
            COMTAM_THROW_ERROR(std::runtime_error, "DType didn't match with input data");
        }

        /*
         *In this case, we trust the caller that the data
         * is valid and length of data is equal to the number of elements in the
         * view
         */

        view view_value = view(shape);
        auto device = runtime_state->device();

        // create a storage
        auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_int>(view_value.numel()));
        device.copy<scalar_t>(data, static_cast<size_int>(view_value.numel()), buffer);

        auto storage = std::make_shared<core::storage>(std::move(buffer));

        return std::make_shared<tensor_impl>(dtype, view_value, std::move(storage),
                                             std::move(runtime_state));
    });
}

template <typename T>
static std::shared_ptr<tensor_impl> make_impl_from_scalar(
    T value, DType dtype, std::shared_ptr<core::runtime_state> runtime_state) {
    return COMTAM_DISPATCH_DTYPE(dtype, [&] {
        if constexpr (!std::is_same_v<T, scalar_t>) {
            COMTAM_THROW_ERROR(std::runtime_error, "DType didn't match with input data");
        }

        view view_value = view({});
        auto device = runtime_state->device();

        auto buffer = device.allocate(sizeof(scalar_t));
        device.copy<scalar_t>(value, buffer);

        auto storage = std::make_shared<core::storage>(std::move(buffer));

        return std::make_shared<tensor_impl>(dtype, view_value, std::move(storage),
                                             std::move(runtime_state));
    });
}

static std::shared_ptr<tensor_impl> make_impl_from_view(
    const view& view, DType dtype, std::shared_ptr<core::runtime_state> runtime_state) {
    return COMTAM_DISPATCH_DTYPE(dtype, [&] {
        auto device = runtime_state->device();
        auto buffer = device.allocate(sizeof(scalar_t) * static_cast<size_int>(view.numel()));

        auto storage = std::make_shared<core::storage>(std::move(buffer));

        return std::make_shared<tensor_impl>(dtype, view, std::move(storage),
                                             std::move(runtime_state));
    });
}

static std::shared_ptr<tensor_impl> make_impl_from_shape(
    const view_vector& shape, DType dtype, std::shared_ptr<core::runtime_state> runtime_state) {
    return make_impl_from_view(view(shape), dtype, std::move(runtime_state));
}

static std::shared_ptr<tensor_impl> make_impl_from_storage(
    std::shared_ptr<core::storage> storage, const view_vector& shape, DType dtype,
    std::shared_ptr<core::runtime_state> runtime_state) {
    return COMTAM_DISPATCH_DTYPE(dtype, [&] {
        view view_value = view(shape);
        COMTAM_CHECK_AND_THROW(
            sizeof(scalar_t) * static_cast<size_int>(view_value.numel()) == storage->size(),
            std::runtime_error, "Storage size doesn't match dtype and shape");

        return std::make_shared<tensor_impl>(dtype, view_value, std::move(storage),
                                             std::move(runtime_state));
    });
}

static std::shared_ptr<tensor_impl> make_impl_from_storage(
    std::shared_ptr<core::storage> storage, const view& view, DType dtype,
    std::shared_ptr<core::runtime_state> runtime_state) {
    /*
     * A view can have a different logical numel from its aliased storage.
     * Readback uses View::physical_offset to map logical indices to storage.
     */
    return std::make_shared<tensor_impl>(dtype, view, std::move(storage), std::move(runtime_state));
}
}  // namespace comtam