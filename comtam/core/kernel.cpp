#include "comtam/core/kernel.h"
#include "Foundation/NSError.hpp"
#include "Foundation/NSSharedPtr.hpp"
#include "Foundation/NSString.hpp"
#include "Metal/MTLDevice.hpp"
#include "comtam/utils/common.h"

#include <filesystem>
#include <stdexcept>

using namespace comtam::core;

namespace fs = std::filesystem;

KernelLibrary::KernelLibrary(MTL::Device* device, fs::path kernel_dir)
    : device_(device)
{
    // we will compile all kernel source at once
    std::string source;
    if (fs::exists(kernel_dir) && fs::is_directory(kernel_dir)) {
        for (const auto& entry : fs::directory_iterator(kernel_dir)) {
            if (entry.path().extension() == ".metal") {
                source += comtam::utils::read_file(entry.path().string());
                source += "\n";
            }
        }
    } else {
        throw std::runtime_error("[ERROR] Directory does not exist.\n");
    }

    NS::Error* error = nullptr;
    auto ns_source = NS::String::string(source.c_str(), NS::UTF8StringEncoding);

    library_ = NS::TransferPtr(device->newLibrary(ns_source, nullptr, &error));

    if (!library_) {
        throw std::runtime_error("[ERROR] Failed to compile Metal kernels: " + comtam::utils::ns_error_message(error));
    }
}

// We only compile source only when we need it
// The name here is the "function" name in kernel source
MTL::ComputePipelineState* KernelLibrary::get(const Op& op) {

    if (auto it = op_to_kernel_name.find(op); it == op_to_kernel_name.end()) {
        throw std::runtime_error("[ERROR] Unsupported operation");
    }

    auto& name = op_to_kernel_name[op];
    if (auto it = pipeline_cache_.find(name); it != pipeline_cache_.end()) {
        return it->second.get();
    }

    auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);

    auto function = NS::TransferPtr(library_->newFunction(ns_name));
    if (!function) {
        throw std::runtime_error("[ERROR] Missing Metal kernel: " + name);
    }

    NS::Error* error = nullptr;
    auto pipeline = NS::TransferPtr(device_->newComputePipelineState(function.get(), &error));

    if (!pipeline) {
        throw std::runtime_error("[ERROR] Failed to create pipeline " + name + ": " +
                                comtam::utils::ns_error_message(error));
    }

    auto* raw = pipeline.get();
    pipeline_cache_.emplace(name, std::move(pipeline));
    return raw;
}
