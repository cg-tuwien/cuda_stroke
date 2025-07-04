cmake_minimum_required(VERSION 3.25)
project(cuda_stroke_lib LANGUAGES CUDA CXX)

alp_add_git_repository(gcem URL https://github.com/kthohr/gcem.git COMMITISH v1.18.0)
alp_add_git_repository(whack URL git@github.com:cg-tuwien/cuda_whack.git COMMITISH 99f607e0dfd1d888fd1722681b2f722c3ccff4cd)

add_library(stroke INTERFACE
    stroke/detail/symmetric_mat.h
    stroke/scalar_functions.h
    stroke/detail/linalg_functions.h
    stroke/pretty_printers.h
    stroke/algorithms.h
    stroke/cuda_compat.h
    stroke/gaussian.h
    stroke/ray.h
    stroke/welford.h
    stroke/detail/symmetric_mat_operators.h
    stroke/geometry.h
    stroke/utility.h
    stroke/unittest/gradcheck.h
    stroke/grad/linalg.h
    stroke/linalg.h
    stroke/grad/util.h
    stroke/grad/scalar_functions.h
    stroke/grad/gaussian.h
    stroke/grad/quaternions.h
    stroke/unittest/random_entity.h
)
target_link_libraries(stroke INTERFACE glm::glm whack gcem)
target_include_directories(stroke INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(stroke INTERFACE GLM_ENABLE_EXPERIMENTAL)
