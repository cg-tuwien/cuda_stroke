cmake_minimum_required(VERSION 3.25)
project(cuda_stroke LANGUAGES CUDA CXX)

option(A_BUILD_STROKE_UNITTESTS "include unit test targets for stroke in the buildsystem" OFF)

include(cmake/AddRepo.cmake)

add_subdirectory(src)

if(A_BUILD_STROKE_UNITTESTS)
    add_subdirectory(unittests)
endif()
