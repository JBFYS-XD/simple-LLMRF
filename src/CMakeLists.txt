add_library(sllmrf
    gguf.cpp
    internlm2.cpp
    operators.cpp
    tensor.cpp
    tokenizer.cpp
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

if (MSVC)
    target_compile_options(sllmrf PRIVATE /W4 /permissive-)
else()
    target_compile_options(sllmrf PRIVATE -Wall -Wextra -Wpedantic)
endif()
