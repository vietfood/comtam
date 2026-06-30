#!/bin/bash

cmake -S . -B build -DCOMTAM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure