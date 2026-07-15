#!/bin/bash

cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON  -DCOMTAM_DEBUG_2=ON -DCMAKE_COLOR_DIAGNOSTICS=ON
cmake --build build
ctest --test-dir build --output-on-failure