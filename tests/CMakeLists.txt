# +--( ~_~ )-------------------------------------------------------------+
# | (c) 2026 Nguyen Le <lenguyen18072003@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                        |
# |                                                                       |
# | Website : https://lenguyen.vercel.app                                 |
# | GitHub  : https://github.com/vietfood/comtam                          |
# | License : https://www.apache.org/licenses/LICENSE-2.0                 |
# +--( ^_^ )-------------------------------------------------------------+

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMTAM_TEST_SOURCES
  core/context.cpp
  core/view.cpp
  tensor/operations.cpp
  tensor/forward.cpp
)

add_executable(comtam_tests
  ${COMTAM_TEST_SOURCES}
)

target_link_libraries(comtam_tests
  PRIVATE
    comtam_lib
    Catch2::Catch2WithMain
    mlxc
)

target_compile_features(comtam_tests PRIVATE cxx_std_20)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/vendor/Catch2/extras")
include(Catch)
catch_discover_tests(comtam_tests)
