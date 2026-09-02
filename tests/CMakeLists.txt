# +--( ~_~ )-------------------------------------------------------------+
# | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
# | Licensed under the MIT License                                        |
# |                                                                       |
# | Website : https://lenguyen.vercel.app                                 |
# | GitHub  : https://github.com/vietfood/comtam                          |
# | License : https://opensource.org/license/mit                          |
# +--( ^_^ )-------------------------------------------------------------+

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMTAM_TEST_SOURCES
  core/context.cpp
  tensor/view.cpp
  tensor/checks.cpp
  tensor/tensor_api.cpp
  tensor/identity.cpp
  tensor/ops_elementwise.cpp
  tensor/ops_broadcast.cpp
  tensor/ops_scalar.cpp
  tensor/ops_reduce.cpp
  tensor/ops_numerics.cpp
  tensor/ops_matmul.cpp
  tensor/movement.cpp
)

add_executable(comtam_tests
  ${COMTAM_TEST_SOURCES}
)

target_link_libraries(comtam_tests
  PRIVATE
    comtam_lib
    Catch2::Catch2WithMain
    mlx
)

target_compile_definitions(comtam_tests
  PRIVATE
    COMTAM_BUILD_TESTS
)

target_compile_features(comtam_tests PRIVATE cxx_std_20)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/vendor/Catch2/extras")
include(Catch)
catch_discover_tests(comtam_tests)
