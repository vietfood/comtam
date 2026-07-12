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

#include <simd/vector_types.h>

#include <string>

#include "comtam/core/storage.h"
#include "comtam/macros/macros.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/op.h"
#include "comtam/tensor/view.h"
#include "comtam/utils/common.h"

namespace comtam::core {
COMTAM_INLINE std::string op2kernel(const Op& op) {
    switch (op) {
        case Op::ADD:
            return "add";
        case Op::SUB:
            return "sub";
        case Op::MUL:
            return "mul";
        case Op::DIV:
            return "div";
        case Op::MATMUL:
            return "matmul";
        case Op::SUM:
            return "reduction_sum";
        case Op::MAX:
            return "reduction_min";
    }
    COMTAM_THROW_ERROR(std::runtime_error, "op isn't supported by kernel");
}

COMTAM_INLINE std::string dtype2kernel(const DType& dtype) {
    switch (dtype) {
        case DType::Float32:
            return "fp32";
    }
    COMTAM_THROW_ERROR(std::runtime_error, "dtype isn't supported by kernel");
}

COMTAM_INLINE bool is_reduce_op(const Op& op) {
    return op == Op::SUM || op == Op::MAX;
}

// a kernel is represented by an op and an dtype
struct kernel_desc {
    Op op;
    DType dtype;
};

// for view and index construction in kernel
struct view_desc {
    size_t N;
    int64_t shape[4];
    int64_t strides[4];
    long offset;
    bool contiguous;

    static view_desc from_view(const view& view) {
        view_desc res;

        res.N = static_cast<size_t>(view.numel());
        res.offset = view.offset;
        res.contiguous = view.is_contiguous();

        // construct shape and strides
        auto shape = utils::arr4_from_vec(view.shape);
        auto strides = utils::arr4_from_vec(view.strides);

        std::copy(shape.begin(), shape.end(), res.shape);
        std::copy(strides.begin(), strides.end(), res.strides);

        return res;
    }
};

// each input will have data buffer
// and view information
struct input_desc {
    storage* storage;
    view_desc view;
};

// A command will have
// - a kernel  (Op + DType)
// - two input info (a, b)
// - an output info (out)
// Warning: we assume this is BinaryCommand
struct command_desc {
    kernel_desc kernel;
    input_desc a;
    input_desc b;
    storage* out_buffer;
};
}  // namespace comtam::core
