#pragma once

#include <stdexcept>
#include <string>

#include "comtam/core/dtype.h"
#include "comtam/core/storage.h"

namespace comtam::core {
enum class Op {
    // Binary operations
    ADD,
    SUB,
    MUL,  // this is element-wise multiplication
    DIV,
};

inline std::string op2kernel(const Op& op) {
    switch (op) {
        case Op::ADD:
            return "add";
        case Op::SUB:
            return "sub";
        case Op::MUL:
            return "mul";
        case Op::DIV:
            return "div";
    }
    throw std::runtime_error("op isn't supported by kernel");
}

inline std::string dtype2kernel(const DType& dtype) {
    switch (dtype) {
        case DType::Float32:
            return "fp32";
    }
    throw std::runtime_error("dtype isn't supported by kernel");
}

// a kernel is represented by an op and an dtype
struct Kernel {
    Op op;
    DType dtype;
};

// A command will have
// - a kernel  (Op + DType)
// - two input Storage pointers (a, b)
// - an output Storage pointer (out)
// - the number of elements to calculate threadgroup
// Warning: we assume this is BinaryCommand
struct Command {
    Kernel kernel;
    Storage* a;
    Storage* b;
    Storage* out;
    size_t elements;
};
}  // namespace comtam::core
