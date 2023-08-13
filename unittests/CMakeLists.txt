cmake_minimum_required(VERSION 3.18)
project(cuda_stroke_unittests LANGUAGES CUDA CXX)

add_executable(stroke_unittests
    main.cpp
    matrix_benchmark.cu
    matrices.cpp
    algorithms.cpp
    gaussian.cpp
)

target_link_libraries(stroke_unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt stroke)
if (UNIX)
    target_compile_options(stroke_unittests PUBLIC -U NDEBUG)
else()
    target_compile_options(stroke_unittests PUBLIC /U NDEBUG)
endif()
