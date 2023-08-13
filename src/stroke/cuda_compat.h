/*****************************************************************************
 * Stroke
 * Copyright (C) 2023 Adam Celarek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#pragma once
#include <cuda_runtime.h>

// #ifdef __CUDACC__
// #warning using nvcc
// #ifdef __CUDA_ARCH__
// #warning device code trajectory
// #else
// #warning nvcc host code trajectory
// #endif
// #else
// #warning non-nvcc code trajectory
// #endif

#ifdef __CUDACC__
#define STROKE_DEVICES __host__ __device__
#else
#define STROKE_DEVICES
#endif

#if defined(NDEBUG)
#if defined(__CUDACC__)
#define STROKE_INLINE __forceinline__
#else
#define STROKE_INLINE inline
#endif
#else
// prevent linker errors for non-template functions
#define STROKE_INLINE static
#endif

#define STROKE_DEVICES_INLINE STROKE_DEVICES STROKE_INLINE
