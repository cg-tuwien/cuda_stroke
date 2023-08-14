cmake_minimum_required(VERSION 3.18)
project(cuda_stroke LANGUAGES CUDA CXX)

option(BUILD_STROKE_UNITTESTS "include unit test targets for stroke in the buildsystem" OFF)

add_subdirectory(src)

if(BUILD_STROKE_UNITTESTS OR BUILD_TESTING)
    add_subdirectory(unittests)
endif()
