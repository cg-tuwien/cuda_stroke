cmake_minimum_required(VERSION 3.18)
project(cuda_stroke_unittests LANGUAGES CUDA CXX)

include(FetchContent)
FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
)
FetchContent_MakeAvailable(catch2)
find_package(CUDAToolkit 12.0 REQUIRED)

add_executable(stroke_unittests
    main.cpp
    matrix_benchmark.cu
    matrices.cpp
    algorithms.cpp
    gaussian.cpp
    welford.cpp
    welford_benchmark.cu
    test_helpers.h
    matrix.cu
    scalar_functions.cpp
    gradcheck.cu
)

target_link_libraries(stroke_unittests PUBLIC Catch2::Catch2WithMain CUDA::nvToolsExt stroke)
target_compile_options(stroke_unittests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
