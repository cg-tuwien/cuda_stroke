cmake_minimum_required(VERSION 3.18)
project(cuda_stroke LANGUAGES CUDA CXX)

option(STROKE_CUDA "include cuda support" ON)
option(STROKE_USE_LLVM_LINKER "use lld (llvm) for linking. it's parallel and much faster, but not installed by default. if it's not installed, you'll get errors, that openmp or other stuff is not installed (hard to track down)" OFF)
option(STROKE_ENABLE_ASSERTS "enable asserts (do not define NDEBUG)" OFF)


############################################ setup ##############################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g -G")

if(NOT MSVC)
    #string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler '-fPIC,-fopenmp,-ffast-math,-fno-finite-math-only,-std=c++17'  --use_fast_math")
#    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --generate-line-info -Xcompiler '-O4,-march=native' -O3") # seems like this stopped working on cuda 11.8 and g++ 11.3
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " --generate-line-info -Xcompiler -O3 -march=native")

    # gcc 10 is buggy (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102)
    #string(APPEND CMAKE_CUDA_FLAGS " -ccbin /usr/bin/g++-9")

    string(APPEND CMAKE_CXX_FLAGS " -ffast-math -fno-finite-math-only")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O4 -march=native")
endif()

if (STROKE_USE_LLVM_LINKER)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=lld")
endif()

######################################## dependencies ###########################################
include(FetchContent)

FetchContent_Declare(whack
    GIT_REPOSITORY  git@github.com:cg-tuwien/cuda_whack.git
    GIT_TAG         origin/main
    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/3rdparty/whack
)
FetchContent_MakeAvailable(whack)

FetchContent_Declare(catch2
    GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
    GIT_TAG         v3.4.0
    SOURCE_DIR 		${CMAKE_SOURCE_DIR}/3rdparty/Catch2
)
FetchContent_MakeAvailable(catch2)

if (STROKE_CUDA)
    find_package(CUDAToolkit)
endif()

########################################## projects ##############################################

add_subdirectory(src)
add_subdirectory(unittests)
