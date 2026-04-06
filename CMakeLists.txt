cmake_minimum_required(VERSION 3.20)

project(simple-LLMRF VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SLLMRF_BUILD_TESTS "Build test targets" ON)
option(SLLMRF_BUILD_TOOLS "Build CLI tools" ON)
option(SLLMRF_ENABLE_OPENMP "Enable OpenMP parallel operators" ON)

include(CTest)
include(ProcessorCount)

ProcessorCount(SLLMRF_HOST_CPU_COUNT)
if (NOT SLLMRF_HOST_CPU_COUNT OR SLLMRF_HOST_CPU_COUNT EQUAL 0)
    set(SLLMRF_HOST_CPU_COUNT 1)
endif()

if (SLLMRF_ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message(STATUS "OpenMP support enabled")
    else()
        message(WARNING "OpenMP requested but not found; falling back to single-threaded operators")
    endif()
endif()

add_subdirectory(src)

if (SLLMRF_BUILD_TOOLS)
    add_subdirectory(app)
endif()

if (SLLMRF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
