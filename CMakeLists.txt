cmake_minimum_required(VERSION 3.24)

project(comtam LANGUAGES CXX)

option(COMTAM_BUILD_TESTS "Build comtam tests" OFF)
option(COMTAM_BUILD_METALLIB "Compile Metal kernels into a default.metallib" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  add_subdirectory(tests)
endif()
