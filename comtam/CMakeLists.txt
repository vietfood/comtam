set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(comtam
  core/context.cpp
  core/kernel.cpp
  core/storage.cpp
  core/device.cpp
  main.cpp
)

target_compile_definitions(comtam
  PRIVATE
    COMTAM_KERNEL_DIR="${PROJECT_BINARY_DIR}/kernels"
)

target_include_directories(comtam
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_compile_features(comtam PUBLIC cxx_std_20)

target_link_libraries(comtam
  PRIVATE
    metal_cpp
)

set_target_properties(comtam PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

file(GLOB COMTAM_METAL_KERNELS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal
)

if(COMTAM_METAL_KERNELS)
  target_sources(comtam PRIVATE ${COMTAM_METAL_KERNELS})
  set_source_files_properties(${COMTAM_METAL_KERNELS} PROPERTIES
    HEADER_FILE_ONLY TRUE
  )

  add_custom_target(comtam_kernel_sources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels
      ${PROJECT_BINARY_DIR}/kernels
    DEPENDS ${COMTAM_METAL_KERNELS}
    VERBATIM
  )

  add_dependencies(comtam comtam_kernel_sources)
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
      DEPENDS ${kernel}
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

  add_dependencies(comtam comtam_kernels)
endif()
