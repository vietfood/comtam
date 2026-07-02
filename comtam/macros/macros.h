/*
** +--( ~_~ )-------------------------------------------------------------+
** | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                        |
** | AI assist : Composer 2.5 (Cursor) and GPT 5.5 (Codex)                 |
** |                                                                       |
** | Website : https://lenguyen.vercel.app                                 |
** | GitHub  : https://github.com/vietfood/comtam                          |
** | License : https://www.apache.org/licenses/LICENSE-2.0                 |
** +--( ^_^ )-------------------------------------------------------------+
*/

#pragma once

#define COMTAM_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#if defined(_MSC_VER)
#define COMTAM_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define COMTAM_INLINE __attribute__((__always_inline__)) inline
#else
#define COMTAM_INLINE inline
#endif
