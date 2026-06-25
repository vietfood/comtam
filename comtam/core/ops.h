#pragma once

#include <unordered_map>
#include "comtam/core/storage.h"
#include <string>

namespace comtam::core {
enum class Op {
    // Binary operations
    ADD,
    SUB,
    MUL, // this is element-wise multiplication
    DIV,
};

// for kernel name lookup
static std::unordered_map<Op, std::string> op_to_kernel_name = {
    {Op::ADD, "add"},
    {Op::SUB, "sub"},
    {Op::MUL, "mul"},
    {Op::DIV, "div"},
};

// A command will have
// - an operation (Op)
// - two input Storage pointers (a, b)
// - an output Storage pointer (out)
// - the number of elements to calculate threadgroup
// Warning: we assume this is BinaryCommand
struct Command {
    Op op;
    Storage* a;
    Storage* b;
    Storage* out;
    size_t elements;
};

}
