cmake_minimum_required(VERSION 3.19)
project(cuda_stroke_lib LANGUAGES CUDA CXX)

########################################## projects ##############################################
add_library(stroke INTERFACE
    stroke/matrix.h
    stroke/scalar_functions.h
    stroke/matrix_functions.h
    stroke/pretty_printers.h
    stroke/algorithms.h
    stroke/cuda_compat.h
    stroke/gaussian.h
    stroke/ray.h
    stroke/welford.h
    stroke/matrix_operators.h
)
target_link_libraries(stroke INTERFACE glm::glm whack gcem)
target_include_directories(stroke INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
