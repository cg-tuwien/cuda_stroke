/*****************************************************************************
 * Alpine Terrain Renderer
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

#include <cuda/std/functional>

#include "algorithms.h"
#include "matrix.h"
#include "scalar_functions.h"

namespace stroke {
template <glm::length_t n_dims, typename scalar_t>
struct SymmetricMat;

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE bool operator==(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return a.data() == b.data();
}

// matrix scalar binary functions
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator+(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::plus<scalar_t> {}) };
}
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> operator-(const SymmetricMat<n_dims, scalar_t>& a, const scalar_t& b)
{
    return { cwise_fun(a.data(), b, cuda::std::minus<scalar_t> {}) };
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

// component wise functions
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
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> matrixCompMult(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return { cwise_fun(a.data(), b.data(), cuda::std::multiplies<scalar_t> {}) };
}
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, scalar_t> cwise_mul(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return matrixCompMult(a, b);
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

// unary functions
template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t determinant(const SymmetricMat<3, scalar_t>& m)
{
    return m[0] * m[3] * m[5] + 2 * m[1] * m[2] * m[4] - m[0] * m[4] * m[4] - m[3] * m[2] * m[2] - m[5] * m[1] * m[1];
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t determinant(const SymmetricMat<2, scalar_t>& m)
{
    return m[0] * m[2] - sq(m[1]);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t det(const SymmetricMat<n_dims, scalar_t>& m)
{
    return determinant(m);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const SymmetricMat<2, scalar_t>& m)
{
    return m[0] + 2 * m[1] + m[2];
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const SymmetricMat<3, scalar_t>& m)
{
    return m[0] + 2 * m[1] + 2 * m[2] + m[3] + 2 * m[4] + m[5];
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<2, scalar_t> diagonal(const SymmetricMat<2, scalar_t>& m)
{
    return { m[0], m[2] };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<3, scalar_t> diagonal(const SymmetricMat<3, scalar_t>& m)
{
    return { m[0], m[3], m[5] };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<2, scalar_t> inverse(const SymmetricMat<2, scalar_t>& m)
{
    return (scalar_t(1) / det(m)) * SymmetricMat<2, scalar_t>(m[2], -m[1], m[0]);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE SymmetricMat<3, scalar_t> inverse(const SymmetricMat<3, scalar_t>& m)
{
    const auto i11 = m[3] * m[5] - m[4] * m[4];
    const auto i12 = m[2] * m[4] - m[1] * m[5];
    const auto i13 = m[1] * m[4] - m[2] * m[3];
    const auto i22 = m[0] * m[5] - m[2] * m[2];
    const auto i23 = m[1] * m[2] - m[0] * m[4];
    const auto i33 = m[0] * m[3] - m[1] * m[1];
    return (scalar_t(1) / det(m)) * SymmetricMat<3, scalar_t>(i11, i12, i13, i22, i23, i33);
}

} // namespace stroke

namespace glm {
// convenience functions for the glm namespace
template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t det(const mat<n_dims, n_dims, scalar_t>& m)
{
    return determinant(m);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, scalar_t> cwise_mul(const glm::mat<n_dims, n_dims, scalar_t>& m1, const glm::mat<n_dims, n_dims, scalar_t>& m2)
{
    return matrixCompMult(m1, m2);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const glm::vec<2, scalar_t>& v)
{
    return v[0] + v[1];
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const glm::vec<3, scalar_t>& v)
{
    return v[0] + v[1] + v[2];
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const glm::mat<2, 2, scalar_t>& m)
{
    return sum(m[0]) + sum(m[1]);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sum(const glm::mat<3, 3, scalar_t>& m)
{
    return sum(m[0]) + sum(m[1]) + sum(m[2]);
}

template <int DIMS, typename scalar_t>
STROKE_DEVICES_INLINE glm::vec<DIMS, scalar_t> diagonal(const glm::mat<DIMS, DIMS, scalar_t>& x)
{
    glm::vec<DIMS, scalar_t> d;
    for (unsigned i = 0; i < DIMS; ++i)
        d[i] = x[i][i];
    return d;
}

} // namespace glm
