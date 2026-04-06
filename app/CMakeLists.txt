set(TARGET sllmrf_cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE sllmrf)
target_compile_features(${TARGET} PRIVATE cxx_std_20)
target_compile_definitions(${TARGET} PRIVATE SLLMRF_PROJECT_ROOT="${PROJECT_SOURCE_DIR}")
