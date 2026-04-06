set(SLLMRF_LIBRARY_SOURCES
    device.cpp
    gguf.cpp
    internlm2.cpp
    operators.cpp
    tensor.cpp
    tokenizer.cpp
)

if (SLLMRF_ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND SLLMRF_LIBRARY_SOURCES
        internlm2_cuda.cu
        operators_cuda.cu
    )
endif()

add_library(sllmrf
    ${SLLMRF_LIBRARY_SOURCES}
)

target_include_directories(sllmrf
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(sllmrf PUBLIC cxx_std_20)

if (SLLMRF_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(sllmrf PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(sllmrf PUBLIC SLLMRF_USE_OPENMP=1)
else()
    target_compile_definitions(sllmrf PUBLIC SLLMRF_USE_OPENMP=0)
endif()

if (SLLMRF_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(sllmrf PUBLIC CUDA::cudart)
    target_compile_definitions(sllmrf PUBLIC SLLMRF_USE_CUDA=1)
    set_target_properties(sllmrf PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
else()
    target_compile_definitions(sllmrf PUBLIC SLLMRF_USE_CUDA=0)
endif()

if (MSVC)
    target_compile_options(sllmrf PRIVATE /W4 /permissive-)
else()
    target_compile_options(sllmrf PRIVATE -Wall -Wextra -Wpedantic)
endif()
