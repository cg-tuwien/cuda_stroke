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
#include "stroke/scalar_functions.h"

namespace stroke {
template <glm::length_t n_dims, typename Scalar>
struct SymmetricMat;

// binary functions

/// computes M * S * transpose(M)
template <typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<2, Scalar> affine_transform(const SymmetricMat<2, Scalar>& S, const glm::mat<2, 2, Scalar>& M)
{
    return {
        M[0][0] * (S[0] * M[0][0] + S[1] * M[1][0]) + M[1][0] * (S[1] * M[0][0] + S[2] * M[1][0]),
        M[0][0] * (S[0] * M[0][1] + S[1] * M[1][1]) + M[1][0] * (S[1] * M[0][1] + S[2] * M[1][1]),
        M[0][1] * (S[0] * M[0][1] + S[1] * M[1][1]) + M[1][1] * (S[1] * M[0][1] + S[2] * M[1][1])
    };
}
/// computes M * S * transpose(M)
template <typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<3, Scalar> affine_transform(const SymmetricMat<3, Scalar>& S, const glm::mat<3, 3, Scalar>& M)
{
    const auto MS = M * glm::mat<3, 3, Scalar>(S);
    const auto Mt = transpose(M);
    return SymmetricMat<3, Scalar> { MS * Mt };
    //    return {
    //        M[0][0] * (S[0] * M[0][0] + S[1] * M[1][0] + S[2] * M[2][0]) + M[1][0] * (S[1] * M[0][0] + S[3] * M[1][0] + S[4] * M[2][0]) + M[2][0] * (S[2] * M[0][0] + S[4] * M[1][0] + S[5] * M[2][0]),
    //        M[0][0] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][0] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][0] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1]),
    //        M[0][0] * (S[0] * M[0][2] + S[1] * M[1][2] + S[2] * M[2][2]) + M[1][0] * (S[1] * M[0][2] + S[3] * M[1][2] + S[4] * M[2][2]) + M[2][0] * (S[2] * M[0][2] + S[4] * M[1][2] + S[5] * M[2][2]),
    //        M[0][1] * (S[0] * M[0][1] + S[1] * M[1][1] + S[2] * M[2][1]) + M[1][1] * (S[1] * M[0][1] + S[3] * M[1][1] + S[4] * M[2][1]) + M[2][1] * (S[2] * M[0][1] + S[4] * M[1][1] + S[5] * M[2][1]),
    //        M[0][1] * (S[0] * M[0][2] + S[1] * M[1][2] + S[2] * M[2][2]) + M[1][1] * (S[1] * M[0][2] + S[3] * M[1][2] + S[4] * M[2][2]) + M[2][1] * (S[2] * M[0][2] + S[4] * M[1][2] + S[5] * M[2][2]),
    //        M[0][2] * (S[0] * M[0][2] + S[1] * M[1][2] + S[2] * M[2][2]) + M[1][2] * (S[1] * M[0][2] + S[3] * M[1][2] + S[4] * M[2][2]) + M[2][2] * (S[2] * M[0][2] + S[4] * M[1][2] + S[5] * M[2][2])
    //    };
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, Scalar> matrixCompMult(const SymmetricMat<n_dims, Scalar>& a, const SymmetricMat<n_dims, Scalar>& b)
{
    return { cwise_fun(a.data(), b.data(), cuda::std::multiplies<Scalar> {}) };
}
template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<n_dims, Scalar> cwise_mul(const SymmetricMat<n_dims, Scalar>& a, const SymmetricMat<n_dims, Scalar>& b)
{
    return matrixCompMult(a, b);
}

// unary functions
template <typename Scalar>
STROKE_DEVICES_INLINE Scalar determinant(const SymmetricMat<3, Scalar>& m)
{
    return m[0] * m[3] * m[5] + 2 * m[1] * m[2] * m[4] - m[0] * m[4] * m[4] - m[3] * m[2] * m[2] - m[5] * m[1] * m[1];
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar determinant(const SymmetricMat<2, Scalar>& m)
{
    return m[0] * m[2] - sq(m[1]);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar det(const SymmetricMat<n_dims, Scalar>& m)
{
    return determinant(m);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const SymmetricMat<2, Scalar>& m)
{
    return m[0] + 2 * m[1] + m[2];
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const SymmetricMat<3, Scalar>& m)
{
    return m[0] + 2 * m[1] + 2 * m[2] + m[3] + 2 * m[4] + m[5];
}

template <typename Scalar>
STROKE_DEVICES_INLINE glm::vec<2, Scalar> diagonal(const SymmetricMat<2, Scalar>& m)
{
    return { m[0], m[2] };
}

template <typename Scalar>
STROKE_DEVICES_INLINE glm::vec<3, Scalar> diagonal(const SymmetricMat<3, Scalar>& m)
{
    return { m[0], m[3], m[5] };
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE bool isnan(const SymmetricMat<n_dims, Scalar>& m)
{
    return reduce(m.data(), false, [](bool boolean, const Scalar& v) { return boolean || isnan(v); });
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar trace(const SymmetricMat<n_dims, Scalar>& m)
{
    return sum(diagonal(m));
}

template <typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<2, Scalar> inverse(const SymmetricMat<2, Scalar>& m)
{
    return (Scalar(1) / det(m)) * SymmetricMat<2, Scalar>(m[2], -m[1], m[0]);
}

template <typename Scalar>
STROKE_DEVICES_INLINE SymmetricMat<3, Scalar> inverse(const SymmetricMat<3, Scalar>& m)
{
    const auto i11 = m[3] * m[5] - m[4] * m[4];
    const auto i12 = m[2] * m[4] - m[1] * m[5];
    const auto i13 = m[1] * m[4] - m[2] * m[3];
    const auto i22 = m[0] * m[5] - m[2] * m[2];
    const auto i23 = m[1] * m[2] - m[0] * m[4];
    const auto i33 = m[0] * m[3] - m[1] * m[1];
    return (Scalar(1) / det(m)) * SymmetricMat<3, Scalar>(i11, i12, i13, i22, i23, i33);
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE bool isnan(const glm::vec<DIMS, Scalar>& x)
{
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        nan = nan || isnan(x[i]);
    return nan;
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE glm::vec<DIMS, Scalar> floor(const glm::vec<DIMS, Scalar>& x)
{
    glm::vec<DIMS, Scalar> f;
    for (unsigned i = 0; i < DIMS; ++i)
        f[i] = stroke::floor(x[i]);
    return f;
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE bool isnan(const glm::mat<DIMS, DIMS, Scalar>& x)
{
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        nan = nan || isnan(x[i]);
    return nan;
}

} // namespace stroke

namespace glm {
// convenience functions for the glm namespace
template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar det(const mat<n_dims, n_dims, Scalar>& m)
{
    return determinant(m);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, Scalar> cwise_mul(const glm::mat<n_dims, n_dims, Scalar>& m1, const glm::mat<n_dims, n_dims, Scalar>& m2)
{
    return matrixCompMult(m1, m2);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const glm::vec<2, Scalar>& v)
{
    return v[0] + v[1];
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const glm::vec<3, Scalar>& v)
{
    return v[0] + v[1] + v[2];
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const glm::vec<4, Scalar>& v)
{
    return v[0] + v[1] + v[2] + v[3];
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const glm::mat<2, 2, Scalar>& m)
{
    return sum(m[0]) + sum(m[1]);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar sum(const glm::mat<3, 3, Scalar>& m)
{
    return sum(m[0]) + sum(m[1]) + sum(m[2]);
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE glm::vec<DIMS, Scalar> diagonal(const glm::mat<DIMS, DIMS, Scalar>& x)
{
    glm::vec<DIMS, Scalar> d;
    for (unsigned i = 0; i < DIMS; ++i)
        d[i] = x[i][i];
    return d;
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE bool isnan(const glm::vec<DIMS, Scalar>& x)
{
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        nan = nan || isnan(x[i]);
    return nan;
}

template <int DIMS, typename Scalar>
STROKE_DEVICES_INLINE bool isnan(const glm::mat<DIMS, DIMS, Scalar>& x)
{
    bool nan = false;
    for (unsigned i = 0; i < DIMS; ++i)
        nan = nan || isnan(x[i]);
    return nan;
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar trace(const glm::mat<n_dims, n_dims, Scalar>& m)
{
    return sum(diagonal(m));
}

} // namespace glm
