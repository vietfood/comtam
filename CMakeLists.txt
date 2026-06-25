cmake_minimum_required(VERSION 3.24)

project(comtam LANGUAGES CXX)

option(COMTAM_BUILD_TESTS "Build comtam tests" OFF)
option(COMTAM_BUILD_METALLIB "Compile Metal kernels into a default.metallib" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(vendor/metal)
add_subdirectory(comtam)

if(COMTAM_BUILD_TESTS)
  message(STATUS "Building tests")
  enable_testing()
  add_subdirectory(tests)
endif()
