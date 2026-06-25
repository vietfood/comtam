#pragma once

#include <unordered_map>
#include <string>
#include <type_traits>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLLibrary.hpp"
#include "comtam/core/ops.h"
#include <filesystem>

namespace comtam::core {


class KernelLibrary {
public:
    KernelLibrary(MTL::Device* device, std::filesystem::path kernel_dir);

    MTL::ComputePipelineState* get(const Op& op);

private:
    // kernel will cache its own Device
    MTL::Device* device_;
    NS::SharedPtr<MTL::Library> library_;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pipeline_cache_;
};
}
