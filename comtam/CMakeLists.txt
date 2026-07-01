set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(COMTAM_DEBUG_1 "comtam: enable INFO logs" OFF)
option(COMTAM_DEBUG_2 "comtam: enable DEBUG logs" OFF)
option(COMTAM_DEBUG_3 "comtam: enable TRACE logs" OFF)

set(COMTAM_DEBUG_DEFINITIONS)
if(COMTAM_DEBUG_1)
  message(STATUS "comtam: debug level 1 (INFO)")
  list(APPEND COMTAM_DEBUG_DEFINITIONS COMTAM_DEBUG_1)
endif()
if(COMTAM_DEBUG_2)
  message(STATUS "comtam: debug level 2 (DEBUG)")
  list(APPEND COMTAM_DEBUG_DEFINITIONS COMTAM_DEBUG_2)
endif()
if(COMTAM_DEBUG_3)
  message(STATUS "comtam: debug level 3 (TRACE)")
  list(APPEND COMTAM_DEBUG_DEFINITIONS COMTAM_DEBUG_3)
endif()

add_library(comtam_lib STATIC
  core/context.cpp
  core/kernel.cpp
  core/storage.cpp
  core/device.cpp
  core/view.cpp
  tensor.cpp
)

target_compile_options(comtam_lib PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wswitch-enum>
)

target_compile_definitions(comtam_lib
  PUBLIC
    COMTAM_KERNEL_DIR="${PROJECT_BINARY_DIR}/kernels"
    ${COMTAM_DEBUG_DEFINITIONS}
)

target_include_directories(comtam_lib
  PUBLIC
    ${PROJECT_SOURCE_DIR}
  SYSTEM PUBLIC
    ${CMAKE_SOURCE_DIR}/vendor/fmt/include
)

target_compile_features(comtam_lib PUBLIC cxx_std_20)

target_link_libraries(comtam_lib
  PUBLIC
    fmt::fmt-header-only
    metal_cpp
)

file(GLOB COMTAM_METAL_KERNELS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal
)

file(GLOB COMTAM_METAL_HEADERS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.h
)

if(COMTAM_METAL_KERNELS)
  target_sources(comtam_lib PRIVATE ${COMTAM_METAL_KERNELS})
  set_source_files_properties(${COMTAM_METAL_KERNELS} PROPERTIES
    HEADER_FILE_ONLY TRUE
  )

  add_custom_target(comtam_kernel_sources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels
      ${PROJECT_BINARY_DIR}/kernels
    DEPENDS ${COMTAM_METAL_KERNELS} ${COMTAM_METAL_HEADERS}
    VERBATIM
  )

  add_dependencies(comtam_lib comtam_kernel_sources)
endif()

if(COMTAM_BUILD_METALLIB AND COMTAM_METAL_KERNELS)
  find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

  set(COMTAM_KERNEL_BUILD_DIR ${PROJECT_BINARY_DIR}/kernels)
  set(COMTAM_METALLIB ${COMTAM_KERNEL_BUILD_DIR}/default.metallib)
  set(COMTAM_AIR_FILES)

  foreach(kernel ${COMTAM_METAL_KERNELS})
    get_filename_component(kernel_name ${kernel} NAME_WE)
    set(air_file ${COMTAM_KERNEL_BUILD_DIR}/${kernel_name}.air)

    add_custom_command(
      OUTPUT ${air_file}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${COMTAM_KERNEL_BUILD_DIR}
      COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${kernel} -o ${air_file}
      DEPENDS ${kernel} ${COMTAM_METAL_HEADERS}
      VERBATIM
    )

    list(APPEND COMTAM_AIR_FILES ${air_file})
  endforeach()

  add_custom_command(
    OUTPUT ${COMTAM_METALLIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${COMTAM_AIR_FILES} -o ${COMTAM_METALLIB}
    DEPENDS ${COMTAM_AIR_FILES}
    VERBATIM
  )

  add_custom_target(comtam_kernels ALL
    DEPENDS ${COMTAM_METALLIB}
  )

  add_dependencies(comtam_lib comtam_kernels)
endif()
