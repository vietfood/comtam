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
    const std::string kernel_name = command.kernel.name();
    const comtam::size_int N = command.a.view.N;

    auto* command_buffer = command_queue->commandBuffer();
    COMTAM_CHECK_AND_THROW(command_buffer, std::runtime_error, "Failed to create command buffer");

    auto* encoder = command_buffer->computeCommandEncoder();
    COMTAM_CHECK_AND_THROW(encoder, std::runtime_error, "Failed to create compute command encoder");

    encoder->setComputePipelineState(pipeline);
    std::forward<EncodeFn>(encode)(encoder, pipeline, kernel_name, N);

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
    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            // set storage first
            encoder->setBuffer(command.a.storage->ptr(), 0, 0);
            encoder->setBuffer(command.b.storage->ptr(), 0, 1);
            encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

            // then set view info
            encoder->setBytes(&command.a.view, sizeof(view_desc), 3);
            encoder->setBytes(&command.b.view, sizeof(view_desc), 4);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, "
                "out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(command.a.view),
                utils::format_view_info(command.b.view), command.a.storage->size(),
                command.b.storage->size(), command.out_buffer->size(), sizeof(command.b.view));

            auto w = pipeline->threadExecutionWidth();
            MTL::Size threads_per_group = MTL::Size(w, 1, 1);
            MTL::Size threads = MTL::Size(N, 1, 1);

            COMTAM_LOG_DEBUG(
                "submit dispatch:\nN={}\nthread_width={}\nthreads=({}, {}, {})\n"
                "threads_per_group=({}, {}, {})\n",
                N, w, threads.width, threads.height, threads.depth, threads_per_group.width,
                threads_per_group.height, threads_per_group.depth);

            encoder->dispatchThreads(threads, threads_per_group);
        });
}

void metal_device::submit_uop(const command_desc& command, kernel_library& kernels) {
    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            encoder->setBuffer(command.a.storage->ptr(), 0, 0);
            encoder->setBuffer(command.out_buffer->ptr(), 0, 1);

            encoder->setBytes(&command.a.view, sizeof(view_desc), 2);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nstorage_bytes=(a={}, out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(command.a.view), command.a.storage->size(),
                command.out_buffer->size(), sizeof(command.a.view));

            auto w = pipeline->threadExecutionWidth();
            MTL::Size threads_per_group = MTL::Size(w, 1, 1);
            MTL::Size threads = MTL::Size(N, 1, 1);

            COMTAM_LOG_DEBUG(
                "submit dispatch:\nN={}\nthread_width={}\nthreads=({}, {}, {})\n"
                "threads_per_group=({}, {}, {})\n",
                N, w, threads.width, threads.height, threads.depth, threads_per_group.width,
                threads_per_group.height, threads_per_group.depth);

            encoder->dispatchThreads(threads, threads_per_group);
        });
}

void metal_device::submit_matmul(const command_desc& command, kernel_library& kernels) {
    COMTAM_ASSERT(command.kernel.op == Op::MATMUL, "submit_matmul is for matmul op only");

    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            (void)pipeline;  // threadExecutionWidth is not used; BLOCK_SIZE is a fixed kernel
                             // contract.

            // set storage first
            encoder->setBuffer(command.a.storage->ptr(), 0, 0);
            encoder->setBuffer(command.b.storage->ptr(), 0, 1);
            encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

            // then set view info
            encoder->setBytes(&command.a.view, sizeof(view_desc), 3);
            encoder->setBytes(&command.b.view, sizeof(view_desc), 4);

            // TODO: Needs to be set based on kernel variant
            constexpr NS::UInteger BLOCK_SIZE = 16;
            encoder->setBytes(&BLOCK_SIZE, sizeof(NS::UInteger), 5);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, "
                "out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(command.a.view),
                utils::format_view_info(command.b.view), command.a.storage->size(),
                command.b.storage->size(), command.out_buffer->size(), sizeof(command.b.view));

            auto rows = command.a.view.shape[0];
            auto cols = command.b.view.shape[1];

            // One threadgroup computes one BLOCK_SIZE x BLOCK_SIZE output tile.
            // Grid is sized in tiles, not in hardware SIMD width.
            MTL::Size group_size = MTL::Size(BLOCK_SIZE, BLOCK_SIZE, 1);
            MTL::Size grid_size =
                MTL::Size(CEIL_DIV(cols, BLOCK_SIZE), CEIL_DIV(rows, BLOCK_SIZE), 1);

            COMTAM_LOG_DEBUG(
                "submit dispatch:\nN={}\nblock_size={}\nthreads=({}, {}, {})\n"
                "threads_per_group=({}, {}, {})\n",
                N, BLOCK_SIZE, grid_size.width, grid_size.height, grid_size.depth, group_size.width,
                group_size.height, group_size.depth);

            encoder->dispatchThreadgroups(grid_size, group_size);
        });
}

void metal_device::submit_reduce(const command_desc& command, kernel_library& kernels) {
    COMTAM_ASSERT(is_reduce_op(command.kernel.op), "submit_reduce is for reduce op only");

    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            (void)pipeline;  // REDUCE_BLOCK_SIZE is a fixed kernel contract.

            // set storage first
            encoder->setBuffer(command.a.storage->ptr(), 0, 0);
            encoder->setBuffer(command.out_buffer->ptr(), 0, 1);

            // then set view info
            encoder->setBytes(&command.a.view, sizeof(view_desc), 2);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nstorage_bytes=(a={}, out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(command.a.view), command.a.storage->size(),
                command.out_buffer->size(), sizeof(command.a.view));

            // Must match REDUCE_BLOCK_SIZE in reduction.metal; the tree sweep assumes
            // a power-of-two threadgroup size.
            constexpr NS::UInteger BLOCK_SIZE = 256;
            MTL::Size group_size = MTL::Size(BLOCK_SIZE, 1, 1);

            if (command.kernel.variant == OpVariant::FULL) {
                // One threadgroup reduces the whole tensor to a scalar.
                MTL::Size grid_size = MTL::Size(1, 1, 1);

                COMTAM_LOG_DEBUG(
                    "submit reduce full:\nN={}\nblock_size={}\nthreads=({}, {}, {})\n"
                    "threads_per_group=({}, {}, {})\n",
                    N, BLOCK_SIZE, grid_size.width, grid_size.height, grid_size.depth,
                    group_size.width, group_size.height, group_size.depth);

                encoder->dispatchThreadgroups(grid_size, group_size);
            } else {
                // One threadgroup per output element along the reduced axis.
                COMTAM_ASSERT(command.extra.axis >= 0, "axis reduce requires a non-negative axis");
                COMTAM_ASSERT(
                    command.extra.axis < 4 && command.a.view.shape[command.extra.axis] > 0,
                    "axis reduce axis is out of range or has zero extent");

                int32_t axis32 = static_cast<int32_t>(command.extra.axis);
                encoder->setBytes(&axis32, sizeof(int32_t), 3);

                // out_numel = N / shape[axis]
                int64_t out_numel =
                    static_cast<int64_t>(N) / command.a.view.shape[command.extra.axis];

                MTL::Size grid_size = MTL::Size(static_cast<NS::UInteger>(out_numel), 1, 1);

                COMTAM_LOG_DEBUG(
                    "submit reduce axis:\naxis={}\nout_numel={}\nblock_size={}\nthreads=({}, {}, "
                    "{})\n"
                    "threads_per_group=({}, {}, {})\n",
                    command.extra.axis, out_numel, BLOCK_SIZE, grid_size.width, grid_size.height,
                    grid_size.depth, group_size.width, group_size.height, group_size.depth);

                encoder->dispatchThreadgroups(grid_size, group_size);
            }
        });
}
