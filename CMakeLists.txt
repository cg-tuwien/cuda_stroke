cmake_minimum_required(VERSION 3.18)
project(cuda_stroke LANGUAGES CUDA CXX)


option(BUILD_STROKE_UNITTESTS "include unit test targets for stroke in the buildsystem" OFF)
option(AUTOUPDATE_WHACK_LIBRARY "Keeps whack up-to-date with origin/main, but prevents local edits. Change to OFF if you want to edit whack." OFF)

######################################## dependencies ###########################################
include(FetchContent)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

if(NOT TARGET glm::glm)
    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG origin/master
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/glm
    )
    FetchContent_MakeAvailable(glm)
endif()
if(NOT TARGET gcem)
    FetchContent_Declare(gcem
        GIT_REPOSITORY https://github.com/kthohr/gcem.git
        GIT_TAG origin/master
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/gcem
    )
    FetchContent_MakeAvailable(gcem)
endif()
if(BUILD_STROKE_UNITTESTS OR BUILD_TESTING)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
    )
    FetchContent_MakeAvailable(catch2)
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


########################################## projects ##############################################

add_subdirectory(src)

if(BUILD_STROKE_UNITTESTS OR BUILD_TESTING)
    add_subdirectory(unittests)
endif()
