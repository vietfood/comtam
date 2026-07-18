#!/bin/bash

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=ON
cmake --build build
./build/bin/comtam
