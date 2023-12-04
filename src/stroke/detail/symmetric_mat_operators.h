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

#include <cuda/std/functional>
#include <glm/glm.hpp>

#include "stroke/algorithms.h"

namespace stroke {
template <glm::length_t n_dims, typename scalar_t>
struct SymmetricMat;

// binary operators
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE bool operator==(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return a.data() == b.data();
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator+(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::plus<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator/(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::divides<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator-(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::minus<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator-(const SymmetricMat<n_dims, scalar_t>& a)
{
    return { cwise_fun(a.data(), scalar_t(-1), cuda::std::multiplies<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator*(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::multiplies<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator+(const scalar_t& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(b.data(), a, cuda::std::plus<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator-(const scalar_t& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(b.data(), a, [](const auto& b, const auto& a) { return a - b; }) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator*(const scalar_t& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(b.data(), a, cuda::std::multiplies<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator+(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(a.data(), b.data(), cuda::std::plus<scalar_t> {}) };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator-(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(a.data(), b.data(), cuda::std::minus<scalar_t> {}) };
}

// mat vec mul
template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> operator*(const SymmetricMat<2, scalar_t>& m, const glm::vec<2, scalar_t>& v)
{
    return { m[0] * v.x + m[1] * v.y, m[1] * v.x + m[2] * v.y };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> operator*(const SymmetricMat<3, scalar_t>& m, const glm::vec<3, scalar_t>& v)
{
    return {
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[1] * v.x + m[3] * v.y + m[4] * v.z,
        m[2] * v.x + m[4] * v.y + m[5] * v.z
    };
}

// mat mat mul
template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<2, 2, scalar_t> operator*(const SymmetricMat<2, scalar_t>& A, const SymmetricMat<2, scalar_t>& B)
{
    return {
        A[0] * B[0] + A[1] * B[1], A[1] * B[0] + A[2] * B[1],
        A[0] * B[1] + A[1] * B[2], A[1] * B[1] + A[2] * B[2]
    };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> operator*(const SymmetricMat<3, scalar_t>& A, const SymmetricMat<3, scalar_t>& B)
{
    return {
        A[0] * B[0] + A[1] * B[1] + A[2] * B[2],
        A[1] * B[0] + A[3] * B[1] + A[4] * B[2],
        A[2] * B[0] + A[4] * B[1] + A[5] * B[2],

        A[0] * B[1] + A[1] * B[3] + A[2] * B[4],
        A[1] * B[1] + A[3] * B[3] + A[4] * B[4],
        A[2] * B[1] + A[4] * B[3] + A[5] * B[4],

        A[0] * B[2] + A[1] * B[4] + A[2] * B[5],
        A[1] * B[2] + A[3] * B[4] + A[4] * B[5],
        A[2] * B[2] + A[4] * B[4] + A[5] * B[5],
    };
}

// operator assignment (+=, *= etc)
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t>& operator+=(SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    transform_inplace(&a.data(), [b](const auto& a) { return a + b; });
    return a;
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t>& operator*=(SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    transform_inplace(&a.data(), [b](const auto& a) { return a * b; });
    return a;
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t>& operator+=(SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    cwise_inplace_fun(&a.data(), b.data(), cuda::std::plus<scalar_t> {});
    return a;
}

} // namespace stroke
