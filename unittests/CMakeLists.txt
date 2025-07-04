cmake_minimum_required(VERSION 3.25)
project(cuda_stroke_unittests LANGUAGES CUDA CXX)

if (NOT TARGET Catch2::Catch2WithMain)
    alp_add_git_repository(catch2 URL https://github.com/catchorg/Catch2.git COMMITISH v3.6.0)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)

add_executable(stroke_unittests
    main.cpp
    matrix_benchmark.cu
    matrices.cpp
    algorithms.cpp
    gaussian.cpp
    welford.cpp
    welford_benchmark.cu
    matrix.cu
    scalar_functions.cpp
    gradcheck.cu
    grad_gaussian.cu
    grad_linalg.cu
    grad_scalar_functions.cu
    grad_quaternions.cu

)

target_link_libraries(stroke_unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt stroke)
target_compile_options(stroke_unittests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
install(TARGETS stroke_unittests)
