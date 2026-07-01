#pragma once

#define COMTAM_FILENAME \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#if defined(_MSC_VER)
#define COMTAM_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define COMTAM_INLINE __attribute__((__always_inline__)) inline
#else
#define COMTAM_INLINE inline
#endif
