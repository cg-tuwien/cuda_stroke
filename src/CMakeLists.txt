cmake_minimum_required(VERSION 3.19)
project(cuda_stroke_lib LANGUAGES CUDA CXX)
option(AUTOUPDATE_WHACK_LIBRARY "Keeps whack up-to-date with origin/main, but prevents local edits. Change to OFF if you want to edit whack." ON)

include(FetchContent)
if(NOT TARGET gcem)
    FetchContent_Declare(gcem
        GIT_REPOSITORY https://github.com/kthohr/gcem.git
        GIT_TAG origin/master
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/gcem
    )
    FetchContent_MakeAvailable(gcem)
endif()

if(AUTOUPDATE_WHACK_LIBRARY)
    FetchContent_Declare(whack
        GIT_REPOSITORY  git@github.com:cg-tuwien/cuda_whack.git
        GIT_TAG         origin/main
        SOURCE_DIR      ${CMAKE_SOURCE_DIR}/3rdparty/whack
        SOURCE_SUBDIR   src
    )
    FetchContent_MakeAvailable(whack)
else()
    FetchContent_Declare(whack
        GIT_REPOSITORY  git@github.com:cg-tuwien/cuda_whack.git
        GIT_TAG         main
        SOURCE_DIR      ${CMAKE_SOURCE_DIR}/3rdparty/whack
    )
    FetchContent_MakeAvailable(whack)
endif()

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
    stroke/geometry.h
)
target_link_libraries(stroke INTERFACE glm::glm whack gcem)
target_include_directories(stroke INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
