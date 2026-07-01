#include "comtam/core/device.h"

#include "Foundation/NSAutoreleasePool.hpp"
#include "Metal/MTLCommandBuffer.hpp"
#include "Metal/MTLCommandEncoder.hpp"

#include "Metal/MTLComputeCommandEncoder.hpp"
#include "comtam/core/kernel.h"
#include "comtam/core/storage.h"
#include "comtam/utils/common.h"

using namespace comtam::core;

Device::Device() {
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device_) {
        throw std::runtime_error("[ERROR] Failed to create default Metal device");
    }

    command_queue_ = NS::TransferPtr(device_->newCommandQueue());
    if (!command_queue_) {
        throw std::runtime_error("[ERROR] Failed to create Metal command queue");
    }
}

Storage Device::allocate(size_t bytes) {
    if (bytes == 0) {
        throw std::runtime_error("[ERROR] Cannot allocate 0 bytes");
    }
    return Storage(bytes, this->get());
}

void Device::submit(const Command& command, KernelLibrary& kernels) {
    // we create pool to autorelease objects later
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    // get pipeline function
    auto* pipeline = kernels.get(command.kernel);

    // create a command buffer to hold command
    auto* command_buffer = command_queue_->commandBuffer();
    if (!command_buffer) {
        throw std::runtime_error("[ERROR] Failed to create command buffer");
    }

    auto* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        throw std::runtime_error("[ERROR] Failed to create compute command encoder");
    }

    // encode the pipeline state object
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(command.a->ptr(), 0, 0);
    encoder->setBuffer(command.b->ptr(), 0, 1);
    encoder->setBuffer(command.out->ptr(), 0, 2);
    encoder->setBytes(&command.elements, sizeof(command.elements), 3);

    auto w = pipeline->threadExecutionWidth();
    MTL::Size threads_per_group = MTL::Size(w, 1, 1);
    MTL::Size threads = MTL::Size(command.elements, 1, 1);

    // encode the compute command.
    encoder->dispatchThreads(threads, threads_per_group);
    encoder->endEncoding();

    // move command to GPU for execution
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->status() == MTL::CommandBufferStatusError) {
      throw std::runtime_error(
          "[ERROR] Metal command buffer failed: " +
          comtam::utils::ns_error_message(command_buffer->error()));
    }
}
