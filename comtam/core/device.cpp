/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** | AI assist : Grok 4.5 (Cursor)                                         |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#include "comtam/core/device.h"

#include <simd/simd.h>

#include <string>
#include <utility>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Foundation/NSTypes.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLCommandEncoder.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "comtam/core/command.h"
#include "comtam/core/kernel.h"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/utils/common.h"
#include "comtam/utils/debug.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

using namespace comtam::core;

template <typename EncodeFn>
void submit_compute(MTL::CommandQueue* command_queue, const command_desc& command,
                    kernel_library& kernels, EncodeFn&& encode) {
    // we create pool to autorelease objects later
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    auto* pipeline = kernels.get(command.kernel);

    auto* command_buffer = command_queue->commandBuffer();
    COMTAM_CHECK_AND_THROW(command_buffer, std::runtime_error, "Failed to create command buffer");

    auto* encoder = command_buffer->computeCommandEncoder();
    COMTAM_CHECK_AND_THROW(encoder, std::runtime_error, "Failed to create compute command encoder");

    encoder->setComputePipelineState(pipeline);
    std::forward<EncodeFn>(encode)(encoder, pipeline);

    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    COMTAM_CHECK_AND_THROW(command_buffer->status() != MTL::CommandBufferStatusError,
                           std::runtime_error, "Metal command buffer failed: {}",
                           comtam::utils::ns_error_message(command_buffer->error()));
}

metal_device::metal_device() {
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    COMTAM_CHECK_AND_THROW(device_, std::runtime_error, "Failed to create default Metal device");

    command_queue_ = NS::TransferPtr(device_->newCommandQueue());
    COMTAM_CHECK_AND_THROW(command_queue_, std::runtime_error,
                           "Failed to create Metal command queue");
}

storage metal_device::allocate(size_int bytes) {
    COMTAM_CHECK_AND_THROW(bytes != 0, std::runtime_error, "Cannot allocate 0 bytes");
    return storage(bytes, this->get());
}

void metal_device::submit_bop(const command_desc& command, kernel_library& kernels) {
    submit_compute(command_queue_.get(), command, kernels, [&](auto* encoder, auto* pipeline) {
        const std::string kernel_name =
            op2kernel(command.kernel.op) + "_" + dtype2kernel(command.kernel.dtype);

        // set storage first
        encoder->setBuffer(command.a.storage->ptr(), 0, 0);
        encoder->setBuffer(command.b.storage->ptr(), 0, 1);
        encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

        // then set view info
        encoder->setBytes(&command.a.view, sizeof(view_desc), 3);
        encoder->setBytes(&command.b.view, sizeof(view_desc), 4);

        COMTAM_LOG_DEBUG(
            "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, out={})\nview_bytes={}\n",
            kernel_name, utils::format_view_info(command.a.view),
            utils::format_view_info(command.b.view), command.a.storage->size(),
            command.b.storage->size(), command.out_buffer->size(), sizeof(command.b.view));

        auto w = pipeline->threadExecutionWidth();
        MTL::Size threads_per_group = MTL::Size(w, 1, 1);
        MTL::Size threads = MTL::Size(command.a.view.N, 1, 1);

        COMTAM_LOG_DEBUG(
            "submit dispatch:\nN={}\nthread_width={}\nthreads=({}, {}, {})\n"
            "threads_per_group=({}, {}, {})\n",
            command.a.view.N, w, threads.width, threads.height, threads.depth,
            threads_per_group.width, threads_per_group.height, threads_per_group.depth);

        encoder->dispatchThreads(threads, threads_per_group);
    });
}

void metal_device::submit_matmul(const command_desc& command, kernel_library& kernels) {
    COMTAM_ASSERT(command.kernel.op == Op::MATMUL, "submit_matmul is for matmul op only");

    submit_compute(command_queue_.get(), command, kernels, [&](auto* encoder, auto* pipeline) {
        const std::string kernel_name =
            op2kernel(command.kernel.op) + "_" + dtype2kernel(command.kernel.dtype);

        // set storage first
        encoder->setBuffer(command.a.storage->ptr(), 0, 0);
        encoder->setBuffer(command.b.storage->ptr(), 0, 1);
        encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

        // then set view info
        encoder->setBytes(&command.a.view, sizeof(view_desc), 3);
        encoder->setBytes(&command.b.view, sizeof(view_desc), 4);

        // we also need to set blocksize
        auto w = pipeline->threadExecutionWidth();
        encoder->setBytes(&w, sizeof(NS::UInteger), 5);

        COMTAM_LOG_DEBUG(
            "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, out={})\nview_bytes={}\n",
            kernel_name, utils::format_view_info(command.a.view),
            utils::format_view_info(command.b.view), command.a.storage->size(),
            command.b.storage->size(), command.out_buffer->size(), sizeof(command.b.view));

        auto M = command.a.view.shape[0];
        auto N = command.b.view.shape[1];

        MTL::Size group_size = MTL::Size(w, w, 1);
        MTL::Size grid_size = MTL::Size(CEIL_DIV(N, w), CEIL_DIV(M, w), 1);

        COMTAM_LOG_DEBUG(
            "submit dispatch:\nN={}\nthread_width={}\nthreads=({}, {}, {})\n"
            "threads_per_group=({}, {}, {})\n",
            command.a.view.N, w, grid_size.width, grid_size.height, grid_size.depth,
            group_size.width, group_size.height, group_size.depth);

        encoder->dispatchThreadgroups(grid_size, group_size);
    });
}

void metal_device::submit_reduce(const command_desc& command, kernel_library& kernels) {
    COMTAM_ASSERT(is_reduce_op(command.kernel.op), "submit_reduce is for reduce op only");
    (void)kernels;
    // TODO
}
