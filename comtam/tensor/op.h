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

namespace comtam {
enum class Op {
    // Binary operations
    ADD,
    SUB,
    MUL,  // this is element-wise multiplication
    DIV,
    // Matmul
    MATMUL,
    // Reduce
    SUM,
    MAX
};

enum class OpVariant {
    NONE,
    FULL,
    AXIS,
    CONTIGUOUS,
    STRIDED,
};
}  // namespace comtam
