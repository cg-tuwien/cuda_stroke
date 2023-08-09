cmake_minimum_required(VERSION 3.19)
if (STROKE_CUDA)
    project(cuda_stroke_lib LANGUAGES CUDA CXX)
else()
    project(cuda_stroke_lib LANGUAGES CXX)
endif()

########################################## projects ##############################################
add_library(stroke INTERFACE
    stroke/matrix.h
    stroke/scalar_functions.h
    stroke/matrix_functions.h
    stroke/pretty_printers.h
)
target_link_libraries(stroke INTERFACE glm::glm whack)
target_include_directories(stroke INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
