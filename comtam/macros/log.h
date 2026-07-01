#pragma once

#include <fmt/format.h>

#include <cassert>
#include <string>

#include "comtam/macros/macros.h"

// clang-format off

namespace comtam::macros {
COMTAM_INLINE void vlog(const char* level,
                        const char* file,
                        int line,
                        fmt::string_view fmt,
                        fmt::format_args args) {
    fmt::print("[{}] ({}:{}): {}\n", level, file, line, fmt::vformat(fmt, args));
}

COMTAM_INLINE void log(const char* level, const char* file, int line, const char* fmt_str) {
    vlog(level, file, line, fmt_str, fmt::format_args{});
}

template <typename... T>
COMTAM_INLINE void log(const char* level,
                       const char* file,
                       int line,
                       fmt::format_string<T...> fmt,
                       T&&... args) {
    vlog(level, file, line, fmt, fmt::make_format_args(args...));
}
template <typename... T>
COMTAM_INLINE std::string format_message(fmt::format_string<T...> fmt, T&&... args) {
    return fmt::format(fmt, std::forward<T>(args)...);
}

COMTAM_INLINE std::string format_message(const char* msg) { return msg; }
}  // namespace comtam::macros

#define COMTAM_LOG_TEMPLATE(level, fmt_str, ...) \
    comtam::macros::log(level, COMTAM_FILENAME, __LINE__, fmt_str __VA_OPT__(, ) __VA_ARGS__)

#if defined(COMTAM_DEBUG_1)
#define COMTAM_LOG_INFO(fmt_str, ...) COMTAM_LOG_TEMPLATE("info", fmt_str __VA_OPT__(, ) __VA_ARGS__)
#else
#define COMTAM_LOG_INFO(fmt_str, ...)
#endif

#if defined(COMTAM_DEBUG_2)
#define COMTAM_LOG_DEBUG(fmt_str, ...) COMTAM_LOG_TEMPLATE("debug", fmt_str __VA_OPT__(, ) __VA_ARGS__)
#else
#define COMTAM_LOG_DEBUG(fmt_str, ...)
#endif

#if defined(COMTAM_DEBUG_3)
#define COMTAM_LOG_TRACE(fmt_str, ...) COMTAM_LOG_TEMPLATE("trace", fmt_str __VA_OPT__(, ) __VA_ARGS__)
#else
#define COMTAM_LOG_TRACE(fmt_str, ...)
#endif

#define COMTAM_LOG_WARN(fmt_str, ...) COMTAM_LOG_TEMPLATE("warning", fmt_str __VA_OPT__(, ) __VA_ARGS__)
#define COMTAM_LOG_ERR(fmt_str, ...) COMTAM_LOG_TEMPLATE("error", fmt_str __VA_OPT__(, ) __VA_ARGS__)

#define COMTAM_THROW_ERROR(throw_type, fmt_str, ...)                               \
    do {                                                                           \
        COMTAM_LOG_ERR(fmt_str __VA_OPT__(, ) __VA_ARGS__);                        \
        throw throw_type(comtam::macros::format_message(fmt_str __VA_OPT__(, ) __VA_ARGS__)); \
    } while (false)

#define COMTAM_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

#ifdef NDEBUG
#define COMTAM_ASSERT(cond, fmt_str, ...) (void)(cond)
#else
#define COMTAM_ASSERT(cond, fmt_str, ...)                     \
    do {                                                      \
        if (!(cond)) {                                        \
            COMTAM_LOG_ERR(fmt_str __VA_OPT__(, ) __VA_ARGS__); \
            assert(cond);                                     \
        }                                                     \
    } while (false)
#endif

#define COMTAM_CHECK_AND_THROW(cond, throw_type, fmt_str, ...) \
    do {                                                       \
        if (!(cond)) {                                         \
            COMTAM_THROW_ERROR(throw_type, fmt_str __VA_OPT__(, ) __VA_ARGS__); \
        }                                                      \
    } while (false)
