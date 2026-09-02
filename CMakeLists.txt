# +--( ~_~ )-------------------------------------------------------------+
# | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
# | Licensed under the MIT License                                        |
# |                                                                       |
# | Website : https://lenguyen.vercel.app                                 |
# | GitHub  : https://github.com/vietfood/comtam                          |
# | License : https://opensource.org/license/mit                          |
# +--( ^_^ )-------------------------------------------------------------+

cmake_minimum_required(VERSION 3.24)

project(comtam LANGUAGES CXX)

option(COMTAM_BUILD_TESTS "Build comtam tests" OFF)
option(COMTAM_BUILD_METALLIB "Compile Metal kernels into a default.metallib" ON)
option(COMTAM_ENABLE_SANITIZERS
  "Build comtam with AddressSanitizer and UndefinedBehaviorSanitizer (host only)"
  OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Host-side sanitizers for comtam targets. Does not sanitize Metal GPU memory.
add_library(comtam_sanitizers INTERFACE)
if(COMTAM_ENABLE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    message(FATAL_ERROR
      "COMTAM_ENABLE_SANITIZERS requires AppleClang, Clang, or GNU "
      "(got ${CMAKE_CXX_COMPILER_ID})")
  endif()
  message(STATUS
    "comtam: enabling AddressSanitizer + UndefinedBehaviorSanitizer "
    "(host CPU only; Metal GPU buffers are not instrumented)")
  target_compile_options(comtam_sanitizers INTERFACE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -g
  )
  target_link_options(comtam_sanitizers INTERFACE
    -fsanitize=address,undefined
  )
endif()

set(FMT_MASTER_PROJECT OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/fmt EXCLUDE_FROM_ALL)

add_subdirectory(vendor/metal)
add_subdirectory(comtam)

add_executable(comtam main.cpp)
target_link_libraries(comtam PRIVATE comtam_lib)
target_compile_features(comtam PRIVATE cxx_std_20)
set_target_properties(comtam PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

if(COMTAM_BUILD_TESTS)
  message(STATUS "Building tests")
  enable_testing()
  add_subdirectory(vendor/Catch2)

  find_package(MLX REQUIRED)
  message(STATUS "Testing against installed MLX ${MLX_VERSION}")

  add_subdirectory(tests)
endif()
