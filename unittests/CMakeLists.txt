cmake_minimum_required(VERSION 3.18)
project(cuda_stroke_unittests LANGUAGES CUDA CXX)

add_executable(unittests
    main.cpp
    matrix_benchmark.cu
)

target_link_libraries(unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt stroke)
if (UNIX)
    target_compile_options(unittests PUBLIC -U NDEBUG)
else()
    target_compile_options(unittests PUBLIC /U NDEBUG)
endif()
