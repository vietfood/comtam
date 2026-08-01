#!/bin/bash

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCOMTAM_BUILD_TESTS=ON \
  -DCOMTAM_DEBUG_2=ON \
  -DCMAKE_COLOR_DIAGNOSTICS=ON \
  -DCOMTAM_ENABLE_SANITIZERS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
