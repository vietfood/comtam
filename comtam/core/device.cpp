#include "comtam/core/device.h"

#include <string>

#include "Foundation/NSAutoreleasePool.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLCommandEncoder.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "comtam/core/kernel.h"
#include "comtam/core/ops.h"
#include "comtam/core/storage.h"
#include "comtam/macros/log.h"
#include "comtam/utils/common.h"
#include "comtam/utils/debug.h"
#include <simd/simd.h>

using namespace comtam::core;

Device::Device() {
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    COMTAM_CHECK_AND_THROW(device_, std::runtime_error, "Failed to create default Metal device");

    command_queue_ = NS::TransferPtr(device_->newCommandQueue());
    COMTAM_CHECK_AND_THROW(command_queue_, std::runtime_error,
                           "Failed to create Metal command queue");
}

Storage Device::allocate(size_t bytes) {
    COMTAM_CHECK_AND_THROW(bytes != 0, std::runtime_error, "Cannot allocate 0 bytes");
    return Storage(bytes, this->get());
}

void Device::submit(const Command& command, KernelLibrary& kernels) {
    // we create pool to autorelease objects later
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    const std::string kernel_name =
        op2kernel(command.kernel.op) + "_" + dtype2kernel(command.kernel.dtype);

    auto* pipeline = kernels.get(command.kernel);

    // create a command buffer to hold command
    auto* command_buffer = command_queue_->commandBuffer();
    COMTAM_CHECK_AND_THROW(command_buffer, std::runtime_error, "Failed to create command buffer");

    auto* encoder = command_buffer->computeCommandEncoder();
    COMTAM_CHECK_AND_THROW(encoder, std::runtime_error,
                           "Failed to create compute command encoder");

    // encode the pipeline state object
    encoder->setComputePipelineState(pipeline);

    // set storage first
    encoder->setBuffer(command.a.storage->ptr(), 0, 0);
    encoder->setBuffer(command.b.storage->ptr(), 0, 1);
    encoder->setBuffer(command.out_buffer->ptr(), 0, 2);

    // then set view info
    encoder->setBytes(&command.a.view, sizeof(ViewInfo), 3);
    encoder->setBytes(&command.b.view, sizeof(ViewInfo), 4);

    COMTAM_LOG_DEBUG("submit kernel={}\na: {}\nb: {}\nstorage_bytes=(a={}, b={}, out={})\nview_bytes={}\n",
                     kernel_name, utils::format_view_info(command.a.view),
                     utils::format_view_info(command.b.view), command.a.storage->size(),
                     command.b.storage->size(), command.out_buffer->size(), sizeof(command.b.view));


    auto w = pipeline->threadExecutionWidth();
    MTL::Size threads_per_group = MTL::Size(w, 1, 1);
    MTL::Size threads = MTL::Size(command.a.view.N, 1, 1);

    COMTAM_LOG_DEBUG("submit dispatch:\nN={}\nthread_width={}\nthreads=({}, {}, {})\n"
                     "threads_per_group=({}, {}, {})\n",
                     command.a.view.N, w, threads.width, threads.height, threads.depth,
                     threads_per_group.width, threads_per_group.height, threads_per_group.depth);

    encoder->dispatchThreads(threads, threads_per_group);
    encoder->endEncoding();

    // move command to GPU for execution
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    COMTAM_CHECK_AND_THROW(command_buffer->status() != MTL::CommandBufferStatusError,
                           std::runtime_error, "Metal command buffer failed: {}",
                           comtam::utils::ns_error_message(command_buffer->error()));
}
