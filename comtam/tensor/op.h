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

namespace comtam {
enum class Op {
    // Binary operations
    ADD,
    MUL,
    // Unary operations
    NEG,
    RECIP,
    // Matmul
    MATMUL,
    // Reduce
    SUM,
    MAX,
};

enum class OpVariant {
    NONE,
    FULL,
    AXIS,
    CONTIGUOUS,
    STRIDED,
};
}  // namespace comtam
