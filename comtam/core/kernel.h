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

#include <filesystem>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Foundation/NSSharedPtr.hpp"
#include "Metal/MTLComputePipeline.hpp"
#include "Metal/MTLLibrary.hpp"
#include "comtam/core/ops.h"

namespace comtam::core {

class KernelLibrary {
   public:
    KernelLibrary(MTL::Device* device, std::filesystem::path kernel_dir);

    MTL::ComputePipelineState* get(const Kernel& kernel);

   private:
    // kernel will cache its own Device
    MTL::Device* device_;
    NS::SharedPtr<MTL::Library> library_;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pipeline_cache_;
};
}  // namespace comtam::core
