#pragma once

#include <simd/vector_types.h>
#include <string>

#include "comtam/core/dtype.h"
#include "comtam/core/storage.h"
#include "comtam/core/view.h"
#include "comtam/macros/macros.h"
#include "comtam/utils/common.h"

namespace comtam {
// forward decleration
class Tensor;

namespace core {
enum class Op {
    // Binary operations
    ADD,
    SUB,
    MUL,  // this is element-wise multiplication
    DIV,
    // Matmul
    MATMUL
};

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

// a kernel is represented by an op and an dtype
struct Kernel {
    Op op;
    DType dtype;
};

// for view and index construction in kernel
struct ViewInfo {
    size_t N;
    int64_t shape[4];
    int64_t strides[4];
    long offset;
    bool contiguous;

    static ViewInfo from_view(const View& view) {
        ViewInfo res;

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
struct InputInfo {
    Storage* storage;
    ViewInfo view;
};

// A command will have
// - a kernel  (Op + DType)
// - two input info (a, b)
// - an output info (out)
// Warning: we assume this is BinaryCommand
struct Command {
    Kernel kernel;
    InputInfo a;
    InputInfo b;
    Storage* out_buffer;
};
}  // namespace core
}  // namespace comtam
