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

#include "cuda_compat.h"
#include <cmath>
#include <cstdint>

namespace stroke {

// functions that are not in the library
template <typename T>
STROKE_DEVICES_INLINE T sq(const T& v)
{
    return v * v;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE int sign(scalar_t v)
{
    return v >= 0 ? 1 : -1;
}

namespace approximate {
    // http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
    // https://github.com/xodobox/fastapprox/blob/master/fastapprox/src/fastexp.h
    // 2x faster, error in the range of e^-4 (dunno about relativ error)
    STROKE_DEVICES_INLINE float fasterpow2(float p)
    {
        float clipp = (p < -126) ? -126.0f : p;
        union {
            uint32_t i;
            float f;
        } v = { uint32_t((1 << 23) * (clipp + 126.94269504f)) };
        return v.f;
    }

    STROKE_DEVICES_INLINE float fasterexp(float p)
    {
        return fasterpow2(1.442695040f * p);
    }

    // slightly faster than std::exp, slightly less precise (error in the range of e-10)
    STROKE_DEVICES_INLINE float fastpow2(float p)
    {
        float offset = (p < 0) ? 1.0f : 0.0f;
        float clipp = (p < -126) ? -126.0f : p;
        int w = int(clipp);
        float z = clipp - float(w) + offset;
        union {
            uint32_t i;
            float f;
        } v = { uint32_t((1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z)) };

        return v.f;
    }

    STROKE_DEVICES_INLINE float fastexp(float p)
    {
        return fastpow2(1.442695040f * p);
    }

} // namespace approximate

// host and device versions of cmath functions
#ifdef __CUDA_ARCH__
// device versions
__forceinline__ __device__ float max(float a, float b)
{
    return ::fmaxf(a, b);
}
__forceinline__ __device__ double max(double a, double b)
{
    return ::fmax(a, b);
}
template <typename T>
__forceinline__ __device__ T max(T a, T b)
{
    return ::max(a, b);
}
__forceinline__ __device__ float min(float a, float b)
{
    return ::fminf(a, b);
}
__forceinline__ __device__ double min(double a, double b)
{
    return ::fmin(a, b);
}
template <typename T>
__forceinline__ __device__ T min(T a, T b)
{
    return ::min(a, b);
}
__forceinline__ __device__ float exp(float x)
{
    return ::expf(x);
}
__forceinline__ __device__ double exp(double x)
{
    return ::exp(x);
}
__forceinline__ __device__ float pow(float x, float y)
{
    auto v = ::powf(x, y);
    // i'm leaving this assert in, as it can help finding surprising NaNs.
    // if fast math is in place, pow(0, 0) will give a NaN.
    // adding a small epsilon on x helps.
    assert(!::isnan(v));
    return v;
}
__forceinline__ __device__ double pow(double x, double y)
{
    auto v = ::pow(x, y);
    assert(!::isnan(v));
    return v;
}

__forceinline__ __device__ float log(float x)
{
    return ::logf(x);
}
__forceinline__ __device__ double log(double x)
{
    return ::log(x);
}

__forceinline__ __device__ float sqrt(float x)
{
    return ::sqrtf(x);
}
__forceinline__ __device__ double sqrt(double x)
{
    return ::sqrt(x);
}

__forceinline__ __device__ float abs(float x)
{
    return ::fabsf(x);
}
__forceinline__ __device__ double abs(double x)
{
    return ::fabs(x);
}

__forceinline__ __device__ float acos(float x)
{
    return ::acosf(x);
}
__forceinline__ __device__ double acos(double x)
{
    return ::acos(x);
}

__forceinline__ __device__ float cos(float x)
{
    return ::cosf(x);
}
__forceinline__ __device__ double cos(double x)
{
    return ::cos(x);
}

template <typename T>
__forceinline__ __device__ bool isnan(T x)
{
    return ::isnan(x);
}

#else // __CUDA_ARCH__
// host versions. __host__ __device__ are put here only to remove warnings (won't compile in device code)
template <typename scalar_t>
inline __host__ __device__ scalar_t max(scalar_t a, scalar_t b)
{
    return std::max(a, b);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t min(scalar_t a, scalar_t b)
{
    return std::min(a, b);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t exp(scalar_t x)
{
    return std::exp(x);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t pow(scalar_t x, scalar_t y)
{
    return std::pow(x, y);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t log(scalar_t x)
{
    return std::log(x);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t sqrt(scalar_t x)
{
    return std::sqrt(x);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t abs(scalar_t x)
{
    return std::abs(x);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t acos(scalar_t x)
{
    return std::acos(x);
}

template <typename scalar_t>
inline __host__ __device__ scalar_t cos(scalar_t x)
{
    return std::cos(x);
}

template <typename T>
__host__ __device__ inline bool isnan(T x)
{
    return std::isnan(x);
}

#endif

} // namespace stroke
