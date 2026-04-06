set(TARGET sllmrf-test)
add_executable(${TARGET} test.cpp)
target_link_libraries(${TARGET} PRIVATE sllmrf)
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_compile_definitions(${TARGET} PRIVATE SLLMRF_PROJECT_ROOT="${PROJECT_SOURCE_DIR}")

add_test(NAME ${TARGET} COMMAND ${TARGET})
set_tests_properties(
    ${TARGET}
    PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${SLLMRF_HOST_CPU_COUNT};OMP_DYNAMIC=FALSE"
)
