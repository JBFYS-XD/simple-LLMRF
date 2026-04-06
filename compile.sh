#!/usr/bin/env bash

set -euo pipefail

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"

cmake -S . -B build -DSLLMRF_ENABLE_CUDA=ON
cmake --build build --parallel "${CPU_COUNT}"
OMP_NUM_THREADS="${CPU_COUNT}" ctest --test-dir build --output-on-failure --parallel "${CPU_COUNT}"
