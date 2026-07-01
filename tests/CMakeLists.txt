set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMTAM_TEST_SOURCES
  core/context.cpp
  core/view.cpp
  tensor/v0.cpp
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
