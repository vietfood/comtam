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

#include "comtam/core/kernel.h"

#include <filesystem>

#include "Foundation/NSError.hpp"
#include "Foundation/NSSharedPtr.hpp"
#include "Foundation/NSString.hpp"
#include "Metal/MTLDevice.hpp"
#include "comtam/core/command.h"
#include "comtam/macros/log.h"
#include "comtam/utils/common.h"

using namespace comtam::core;

namespace fs = std::filesystem;

kernel_library::kernel_library(MTL::Device* device, fs::path kernel_dir) : device_(device) {
    COMTAM_CHECK_AND_THROW(fs::exists(kernel_dir) && fs::is_directory(kernel_dir),
                           std::runtime_error, "Directory does not exist");

    auto metallib_path = kernel_dir / "default.metallib";
    COMTAM_CHECK_AND_THROW(fs::exists(metallib_path) && fs::is_regular_file(metallib_path),
                           std::runtime_error, "Missing Metal library: {}", metallib_path.string());

    NS::Error* error = nullptr;
    auto ns_path = NS::String::string(metallib_path.string().c_str(), NS::UTF8StringEncoding);

    library_ = NS::TransferPtr(device->newLibrary(ns_path, &error));

    COMTAM_CHECK_AND_THROW(library_, std::runtime_error, "Failed to load Metal library: {}",
                           comtam::utils::ns_error_message(error));
}

// We only compile source only when we need it
// The name here is the "function" name in kernel source
MTL::ComputePipelineState* kernel_library::get(const kernel_desc& kernel, bool is_scalar) {
    std::string name = kernel.name() + (is_scalar ? "_scalar" : "");

    if (auto it = pipeline_cache_.find(name); it != pipeline_cache_.end()) {
        return it->second.get();
    }

    auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);

    auto function = NS::TransferPtr(library_->newFunction(ns_name));
    COMTAM_CHECK_AND_THROW(function, std::runtime_error, "Missing Metal kernel: {}", name);

    NS::Error* error = nullptr;
    auto pipeline = NS::TransferPtr(device_->newComputePipelineState(function.get(), &error));

    COMTAM_CHECK_AND_THROW(pipeline, std::runtime_error, "Failed to create pipeline {}: {}", name,
                           comtam::utils::ns_error_message(error));

    auto* raw = pipeline.get();
    pipeline_cache_.emplace(name, std::move(pipeline));
    return raw;
}
