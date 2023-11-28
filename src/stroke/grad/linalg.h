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

#include <cuda/std/tuple>
#include <glm/glm.hpp>

#include "stroke/cuda_compat.h"
#include "stroke/linalg.h"
#include "util.h"

namespace stroke::grad {
namespace {
    // transpose of adjugate matrix, needed for grad_determinant, which is the cofactor matrix
    // https://en.wikipedia.org/wiki/Jacobi%27s_formula
    // https://en.wikipedia.org/wiki/Adjugate_matrix
    template <typename scalar_t>
    STROKE_DEVICES_INLINE glm::mat<2, 2, scalar_t> cofactor(const glm::mat<2, 2, scalar_t>& m)
    {
        return { { m[1][1], -m[1][0] }, { -m[0][1], m[0][0] } };
    }

    // reduces nicely: https://godbolt.org/z/M5MhET
    template <typename scalar_t>
    STROKE_DEVICES_INLINE scalar_t detexcl(const glm::mat<3, 3, scalar_t>& m, unsigned excl_i, unsigned excl_j)
    {
        // map 0 -> 1, 2 in bits 00 -> 01, 10
        //     1 -> 0, 2         01 -> 00, 10
        //     2 -> 0, 1         10 -> 00, 01
        const auto i1 = unsigned(excl_i < 1);
        const auto i2 = 2 - (excl_i >> 1);

        // same again
        const auto j1 = unsigned(excl_j < 1);
        const auto j2 = 2 - (excl_j >> 1);
        return m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1];
    }
    template <typename scalar_t>
    STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> cofactor(const glm::mat<3, 3, scalar_t>& m)
    {
        glm::mat<3, 3, scalar_t> cof;
        for (unsigned i = 0; i < 3; ++i) {
            for (unsigned j = 0; j < 3; ++j) {
                const auto sign = ((i ^ j) % 2 == 0) ? 1 : -1;
                cof[i][j] = sign * detexcl(m, i, j);
            }
        }
        return cof;
    }
} // namespace

template <typename scalar_t>
STROKE_DEVICES_INLINE stroke::SymmetricMat<3, scalar_t> from_mat_gradient(const glm::mat<3, 3, scalar_t>& g)
{
    constexpr auto half = scalar_t(1.0);
    return {
        g[0][0], half * (g[0][1] + g[1][0]), half * (g[0][2] + g[2][0]),
        g[1][1], half * (g[1][2] + g[2][1]),
        g[2][2]
    };
}
template <typename scalar_t>
STROKE_DEVICES_INLINE glm::mat<3, 3, scalar_t> from_symmetric_gradient(const stroke::SymmetricMat<3, scalar_t>& g)
{
    constexpr auto half = scalar_t(0.5);
    // clang-format off
    return {
        g[0],           half * g[1],    half * g[2],
        half * g[1],           g[3],    half * g[4],
        half * g[2],    half * g[4],           g[5]
    };
    // clang-format on
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, scalar_t> det(const glm::mat<n_dims, n_dims, scalar_t>& mat, scalar_t grad)
{
    return cofactor(mat) * grad;
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::vec<n_dims, scalar_t>, glm::vec<n_dims, scalar_t>> dot(const glm::vec<n_dims, scalar_t>& a, const glm::vec<n_dims, scalar_t>& b, scalar_t incoming_grad)
{
    return { b * incoming_grad, a * incoming_grad };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE glm::vec<n_dims, scalar_t> length(const glm::vec<n_dims, scalar_t>& vec, scalar_t grad)
{
    const auto l = grad / length(vec);
    return vec * l;
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE glm::vec<n_dims, scalar_t> length(const glm::vec<n_dims, scalar_t>& vec, scalar_t grad, scalar_t cached_length)
{
    const auto l = grad / cached_length;
    return vec * l;
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_dims, n_dims, scalar_t>, glm::mat<n_dims, n_dims, scalar_t>>
matmul(const glm::mat<n_dims, n_dims, scalar_t>& a, const glm::mat<n_dims, n_dims, scalar_t>& b, const glm::mat<n_dims, n_dims, scalar_t>& grad)
{
    return { grad * transpose(b), transpose(a) * grad };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<stroke::SymmetricMat<n_dims, scalar_t>, glm::mat<n_dims, n_dims, scalar_t>>
matmul(const stroke::SymmetricMat<n_dims, scalar_t>& a, const glm::mat<n_dims, n_dims, scalar_t>& b, const glm::mat<n_dims, n_dims, scalar_t>& grad)
{
    using mat_t = glm::mat<n_dims, n_dims, scalar_t>;
    return { from_mat_gradient(grad * transpose(b)), mat_t(a) * grad };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_dims, n_dims, scalar_t>, stroke::SymmetricMat<n_dims, scalar_t>>
matmul(const glm::mat<n_dims, n_dims, scalar_t>& a, const stroke::SymmetricMat<n_dims, scalar_t>& b, const glm::mat<n_dims, n_dims, scalar_t>& grad)
{
    using mat_t = glm::mat<n_dims, n_dims, scalar_t>;
    return { grad * mat_t(b), from_mat_gradient(transpose(a) * grad) };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_dims, n_dims, scalar_t>, glm::vec<n_dims, scalar_t>>
matvecmul(const glm::mat<n_dims, n_dims, scalar_t>& a, const glm::vec<n_dims, scalar_t>& b, const glm::vec<n_dims, scalar_t>& grad)
{
    return { outerProduct(grad, b), transpose(a) * grad };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<SymmetricMat<n_dims, scalar_t>, glm::mat<n_dims, n_dims, scalar_t>>
affine_transform(const SymmetricMat<n_dims, scalar_t>& S, const glm::mat<n_dims, n_dims, scalar_t>& M, const stroke::SymmetricMat<n_dims, scalar_t>& grad)
{
    using mat_t = glm::mat<3, 3, scalar_t>;
    const auto MS = M * mat_t(S);
    const auto Mt = transpose(M);
    // return MS * Mt;
    const auto [grad_MS, grad_Mt] = stroke::grad::matmul(MS, Mt, from_symmetric_gradient(grad));

    // const auto MS = M * mat_t(S);
    const auto [grad_M, grad_S] = stroke::grad::matmul(M, S, grad_MS);

    return { grad_S, (grad_M + transpose(grad_Mt)) };
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, scalar_t> inverse_cached(const glm::mat<n_dims, n_dims, scalar_t>& inverse_mat, const glm::mat<n_dims, n_dims, scalar_t>& grad)
{
    // from https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf
    const auto ti = transpose(inverse_mat);
    return -ti * grad * ti;
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, scalar_t> inverse(const glm::mat<n_dims, n_dims, scalar_t>& mat, const glm::mat<n_dims, n_dims, scalar_t>& grad)
{
    return inverse_cached(inverse(mat), grad);
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE stroke::SymmetricMat<n_dims, scalar_t> inverse_cached(const stroke::SymmetricMat<n_dims, scalar_t>& inverse_mat, const stroke::SymmetricMat<n_dims, scalar_t>& grad)
{
    // from https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf
    // ignoring the transpose since we are symmetric
    const auto i = to_glm(inverse_mat);
    return -from_mat_gradient(i * from_symmetric_gradient(grad) * i);
}

template <typename scalar_t, int n_dims>
STROKE_DEVICES_INLINE stroke::SymmetricMat<n_dims, scalar_t> inverse(const stroke::SymmetricMat<n_dims, scalar_t>& mat, const stroke::SymmetricMat<n_dims, scalar_t>& grad)
{
    return inverse_cached(inverse(mat), grad);
}

} // namespace stroke::grad
