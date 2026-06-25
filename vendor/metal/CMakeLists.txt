if(NOT APPLE)
  message(FATAL_ERROR "comtam currently targets Apple platforms because it depends on Metal.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(metal_cpp STATIC metal_cpp.cpp)

target_include_directories(metal_cpp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp
)

target_compile_features(metal_cpp PUBLIC cxx_std_20)

target_link_libraries(metal_cpp
  PUBLIC
    "-framework Foundation"
    "-framework Metal"
)
