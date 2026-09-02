/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the MIT License                                        |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://opensource.org/license/mit                          |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#include "comtam/tensor/tensor.h"

namespace comtam::tests {

// Narrow test-only access for identity claims that must not expose mutable storage.
struct tensor_access {
    static bool shares_storage(const tensor& a, const tensor& b) noexcept {
        return a.impl_ && b.impl_ && a.impl_->storage == b.impl_->storage;
    }

    static bool shares_runtime(const tensor& a, const tensor& b) noexcept {
        return a.impl_ && b.impl_ && a.impl_->runtime_state == b.impl_->runtime_state;
    }

    // Build a distinct logical identity over an existing tensor's storage/runtime.
    // Used for edge-case headers (e.g. zero-extent views) that public factories reject.
    static tensor make_view_alias(const tensor& base, const view& logical_view) {
        auto impl = make_impl_from_storage(base.impl_->storage, logical_view, base.impl_->dtype,
                                           base.impl_->runtime_state);
        return tensor(std::move(impl));
    }
};

}  // namespace comtam::tests
