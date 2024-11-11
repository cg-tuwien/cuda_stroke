/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 ****************************************************************************/

#pragma once

#include "cuda_compat.h"
#include <cmath>
#include <cstdint>

namespace stroke {

template <typename T>
STROKE_DEVICES_INLINE T sq(const T& v)
{
    return v * v;
}

template <typename T>
STROKE_DEVICES_INLINE T cubed(const T& v)
{
    return v * v * v;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE int sign(scalar_t v)
{
    return v >= 0 ? 1 : -1;
}

namespace approximate {
    // static prevents linker errors

    // http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
    // https://github.com/xodobox/fastapprox/blob/master/fastapprox/src/fastexp.h
    // 2x faster, error in the range of e^-4 (dunno about relativ error)
    STROKE_DEVICES_INLINE static float fasterpow2(float p)
    {
        float clipp = (p < -126) ? -126.0f : p;
        union {
            uint32_t i;
            float f;
        } v = { uint32_t((1 << 23) * (clipp + 126.94269504f)) };
        return v.f;
    }

    STROKE_DEVICES_INLINE static float fasterexp(float p)
    {
        return fasterpow2(1.442695040f * p);
    }

    // slightly faster than std::exp, slightly less precise (error in the range of e-10)
    STROKE_DEVICES_INLINE static float fastpow2(float p)
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

    STROKE_DEVICES_INLINE static float fastexp(float p)
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
__forceinline__ __device__ double floor(float x)
{
    return ::floorf(x);
}
__forceinline__ __device__ double floor(double x)
{
    return ::floor(x);
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

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__forceinline__ __device__ bool isnan(T x)
{
    return ::isnan(x);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
__forceinline__ __device__ bool isinf(T x)
{
    return ::isinf(x);
}

__forceinline__ __device__ float erf(float x)
{
    return ::erff(x);
}
__forceinline__ __device__ double erf(double x)
{
    return ::erf(x);
}

__forceinline__ __device__ float ceil(float x)
{
    return ::ceilf(x);
}
__forceinline__ __device__ double ceil(double x)
{
    return ::ceil(x);
}

#else // __CUDA_ARCH__
// host versions, won't be called from device code.
// the __device__ annotations are put here only to remove warnings in qt creator (warning beeing, that host functions can't be called)
template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t max(scalar_t a, scalar_t b)
{
    return std::max(a, b);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t min(scalar_t a, scalar_t b)
{
    return std::min(a, b);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t exp(scalar_t x)
{
    return std::exp(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t pow(scalar_t x, scalar_t y)
{
    return std::pow(x, y);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t log(scalar_t x)
{
    return std::log(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sqrt(scalar_t x)
{
    return std::sqrt(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t abs(scalar_t x)
{
    return std::abs(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t floor(scalar_t x)
{
    return std::floor(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t acos(scalar_t x)
{
    return std::acos(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t cos(scalar_t x)
{
    return std::cos(x);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
STROKE_DEVICES_INLINE bool isnan(T x)
{
    return std::isnan(x);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
STROKE_DEVICES_INLINE bool isinf(T x)
{
    return std::isinf(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t erf(scalar_t x)
{
    return std::erf(x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t ceil(scalar_t x)
{
    return std::ceil(x);
}

#endif

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t clamp(scalar_t v, scalar_t min, scalar_t max)
{
    return stroke::max(min, stroke::min(max, v));
}

} // namespace stroke
