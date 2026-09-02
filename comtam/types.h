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

#include <cstddef>
#include <cstdint>

namespace comtam {
/*
 * Integer aliases by semantic role, not by bit-width.
 *
 * This header is about *index/size* integer types. For *element* data types
 * (float, ...) see comtam/tensor/dtype.h.
 *
 * - view_int: logical domain (shape, stride, offset, axis, coord, numel).
 *   Signed because of -1 sentinels, 0-strides, and reverse loops.
 */
using view_int = int64_t;

/*
 * - size_int: physical domain (byte sizes, element counts into MTL::Buffer,
 *   vector sizes, grid bounds). Aliases size_t; kept for symmetry with
 *   view_int so the boundary cast (view_int -> size_int) reads explicitly.
 */
using size_int = size_t;
}  // namespace comtam
