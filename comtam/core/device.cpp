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
#include <variant>

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

    MTL::ComputePipelineState* pipeline = nullptr;
    std::string kernel_name;
    comtam::size_int N = 0;
    if (command.is_scalar()) {
        auto& cmd = command.as_scalar();
        pipeline = kernels.get(cmd.kernel, true);
        kernel_name = cmd.kernel.name() + "_scalar";
        N = cmd.tensor.view.N;
    } else {
        auto& cmd = command.as_tensor();
        pipeline = kernels.get(cmd.kernel, false);
        kernel_name = cmd.kernel.name();
        N = cmd.a.view.N;
    }

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
            if (command.is_scalar()) {
                auto& cmd = command.as_scalar();
                // set storage
                encoder->setBuffer(cmd.tensor.storage->ptr(), 0, 0);
                encoder->setBuffer(cmd.out_buffer->ptr(), 0, 1);
                // set scalar (right here we assume the scalar type always has 4 bytes)
                encoder->setBytes(&cmd.scalar, sizeof(uint32_t), 2);
                // set view info
                encoder->setBytes(&cmd.tensor.view, sizeof(view_desc), 3);

                COMTAM_LOG_DEBUG(
                    "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, "
                    "out={})\nview_bytes={}\n",
                    kernel_name, utils::format_view_info(cmd.tensor.view), 1,
                    cmd.tensor.storage->size(), sizeof(uint32_t), cmd.out_buffer->size(),
                    sizeof(cmd.tensor.view));
            } else {
                auto& cmd = command.as_tensor();
                // set storage first
                encoder->setBuffer(cmd.a.storage->ptr(), 0, 0);
                encoder->setBuffer(cmd.b.storage->ptr(), 0, 1);
                encoder->setBuffer(cmd.out_buffer->ptr(), 0, 2);

                // then set view info
                encoder->setBytes(&cmd.a.view, sizeof(view_desc), 3);
                encoder->setBytes(&cmd.b.view, sizeof(view_desc), 4);

                COMTAM_LOG_DEBUG(
                    "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, "
                    "out={})\nview_bytes={}\n",
                    kernel_name, utils::format_view_info(cmd.a.view),
                    utils::format_view_info(cmd.b.view), cmd.a.storage->size(),
                    cmd.b.storage->size(), cmd.out_buffer->size(), sizeof(cmd.b.view));
            }

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
            auto& cmd = command.as_tensor();

            encoder->setBuffer(cmd.a.storage->ptr(), 0, 0);
            encoder->setBuffer(cmd.out_buffer->ptr(), 0, 1);

            encoder->setBytes(&cmd.a.view, sizeof(view_desc), 2);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nstorage_bytes=(a={}, out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(cmd.a.view), cmd.a.storage->size(),
                cmd.out_buffer->size(), sizeof(cmd.a.view));

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
    // matmul op is only supported for tensor command
    COMTAM_ASSERT(command.is_tensor(), "submit_matmul is for tensor command only");
    COMTAM_ASSERT(command.as_tensor().kernel.op == Op::MATMUL,
                  "submit_matmul is for matmul op only");

    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            (void)pipeline;  // threadExecutionWidth is not used; BLOCK_SIZE is a fixed kernel
                             // contract.
            auto& cmd = command.as_tensor();

            // set storage first
            encoder->setBuffer(cmd.a.storage->ptr(), 0, 0);
            encoder->setBuffer(cmd.b.storage->ptr(), 0, 1);
            encoder->setBuffer(cmd.out_buffer->ptr(), 0, 2);

            // then set view info
            encoder->setBytes(&cmd.a.view, sizeof(view_desc), 3);
            encoder->setBytes(&cmd.b.view, sizeof(view_desc), 4);

            // TODO: Needs to be set based on kernel variant
            constexpr NS::UInteger BLOCK_SIZE = 16;
            encoder->setBytes(&BLOCK_SIZE, sizeof(NS::UInteger), 5);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, "
                "out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(cmd.a.view),
                utils::format_view_info(cmd.b.view), cmd.a.storage->size(), cmd.b.storage->size(),
                cmd.out_buffer->size(), sizeof(cmd.b.view));

            auto rows = cmd.a.view.shape[0];
            auto cols = cmd.b.view.shape[1];

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
    // reduce op is only supported for tensor command
    COMTAM_ASSERT(command.is_tensor(), "submit_reduce is for tensor command only");
    COMTAM_ASSERT(is_reduce_op(command.as_tensor().kernel.op),
                  "submit_reduce is for reduce op only");

    submit_compute(
        command_queue_.get(), command, kernels,
        [&](auto* encoder, auto* pipeline, const std::string& kernel_name, comtam::size_int N) {
            (void)pipeline;  // REDUCE_BLOCK_SIZE is a fixed kernel contract.
            auto& cmd = command.as_tensor();

            // set storage first
            encoder->setBuffer(cmd.a.storage->ptr(), 0, 0);
            encoder->setBuffer(cmd.out_buffer->ptr(), 0, 1);

            // then set view info
            encoder->setBytes(&cmd.a.view, sizeof(view_desc), 2);

            COMTAM_LOG_DEBUG(
                "submit kernel={}\na: {}\nstorage_bytes=(a={}, out={})\nview_bytes={}\n",
                kernel_name, utils::format_view_info(cmd.a.view), cmd.a.storage->size(),
                cmd.out_buffer->size(), sizeof(cmd.a.view));

            // Must match REDUCE_BLOCK_SIZE in reduction.metal; the tree sweep assumes
            // a power-of-two threadgroup size.
            constexpr NS::UInteger BLOCK_SIZE = 256;
            MTL::Size group_size = MTL::Size(BLOCK_SIZE, 1, 1);

            if (cmd.kernel.variant == OpVariant::FULL) {
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
                COMTAM_ASSERT(cmd.extra.axis >= 0, "axis reduce requires a non-negative axis");
                COMTAM_ASSERT(cmd.extra.axis < 4 && cmd.a.view.shape[cmd.extra.axis] > 0,
                              "axis reduce axis is out of range or has zero extent");

                int32_t axis32 = static_cast<int32_t>(cmd.extra.axis);
                encoder->setBytes(&axis32, sizeof(int32_t), 3);

                // out_numel = N / shape[axis]
                int64_t out_numel = static_cast<int64_t>(N) / cmd.a.view.shape[cmd.extra.axis];

                MTL::Size grid_size = MTL::Size(static_cast<NS::UInteger>(out_numel), 1, 1);

                COMTAM_LOG_DEBUG(
                    "submit reduce axis:\naxis={}\nout_numel={}\nblock_size={}\nthreads=({}, {}, "
                    "{})\n"
                    "threads_per_group=({}, {}, {})\n",
                    cmd.extra.axis, out_numel, BLOCK_SIZE, grid_size.width, grid_size.height,
                    grid_size.depth, group_size.width, group_size.height, group_size.depth);

                encoder->dispatchThreadgroups(grid_size, group_size);
            }
        });
}
