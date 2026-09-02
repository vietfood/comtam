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

#include <string>

#include "comtam/core/storage.h"
#include "comtam/macros/macros.h"
#include "comtam/tensor/dtype.h"
#include "comtam/tensor/op.h"
#include "comtam/tensor/view.h"
#include "comtam/types.h"
#include "comtam/utils/common.h"

namespace comtam::core {
COMTAM_INLINE std::string op2kernel(const Op& op) {
    switch (op) {
        case Op::ADD:
            return "add";
        case Op::MUL:
            return "mul";
        case Op::NEG:
            return "neg";
        case Op::RECIP:
            return "recip";
        case Op::MATMUL:
            return "matmul";
        case Op::SUM:
            return "reduce_sum";
        case Op::MAX:
            return "reduce_max";
    }
    COMTAM_THROW_ERROR(std::runtime_error, "op isn't supported by kernel");
}

COMTAM_INLINE std::string op_variant2kernel(const OpVariant& variant) {
    switch (variant) {
        case OpVariant::FULL:
            return "full";
        case OpVariant::AXIS:
            return "axis";
        case OpVariant::CONTIGUOUS:
            return "contiguous";
        case OpVariant::STRIDED:
            return "strided";
        case OpVariant::NONE:
            return "";
    }
    COMTAM_THROW_ERROR(std::runtime_error, "op variant isn't supported by kernel");
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
    DType dtype = DType::Float32;
    OpVariant variant = OpVariant::NONE;

    std::string name() const {
        return op2kernel(op) + "_" + op_variant2kernel(variant) +
               (variant != OpVariant::NONE ? "_" : "") + dtype2kernel(dtype);
    }
};

/*
 * for view and index construction in kernel
 *
 * Layout must match kernels/utils.h ViewInfo exactly: this struct is sent
 * raw via setBytes and reinterpreted as `constant ViewInfo&` on the GPU side.
 */
struct view_desc {
    size_int N;
    view_int shape[4];
    view_int strides[4];
    view_int offset;
    bool contiguous;

    static view_desc from_view(const view& view) {
        view_desc res;

        res.N = static_cast<size_int>(view.numel());
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

/* each input will have a data buffer and view information */
struct tensor_input_desc {
    storage* storage;
    view_desc view;
};

struct extra_desc {
    view_int axis = -1;
};

/*
 * A command will have:
 * - a kernel  (Op + DType)
 * - two input info (a, b)
 * - an output info (out)
 * - an extra info (axis)
 *
 * Scalars are rank-0 tensors and go through the same path via broadcasting.
 */
struct command_desc {
    kernel_desc kernel;
    tensor_input_desc a;
    tensor_input_desc b;
    storage* out_buffer;
    extra_desc extra;
};

}  // namespace comtam::core
