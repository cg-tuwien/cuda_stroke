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
    template <typename Scalar>
    STROKE_DEVICES_INLINE glm::mat<2, 2, Scalar> cofactor(const glm::mat<2, 2, Scalar>& m)
    {
        return { { m[1][1], -m[1][0] }, { -m[0][1], m[0][0] } };
    }

    // reduces nicely: https://godbolt.org/z/M5MhET
    template <typename Scalar>
    STROKE_DEVICES_INLINE Scalar detexcl(const glm::mat<3, 3, Scalar>& m, unsigned excl_i, unsigned excl_j)
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
    template <typename Scalar>
    STROKE_DEVICES_INLINE glm::mat<3, 3, Scalar> cofactor(const glm::mat<3, 3, Scalar>& m)
    {
        glm::mat<3, 3, Scalar> cof;
        for (unsigned i = 0; i < 3; ++i) {
            for (unsigned j = 0; j < 3; ++j) {
                const auto sign = ((i ^ j) % 2 == 0) ? 1 : -1;
                cof[i][j] = sign * detexcl(m, i, j);
            }
        }
        return cof;
    }
} // namespace

template <typename Scalar>
STROKE_DEVICES_INLINE stroke::SymmetricMat<3, Scalar> to_symmetric_gradient(const glm::mat<3, 3, Scalar>& g)
{
    return {
        g[0][0], g[0][1] + g[1][0], g[0][2] + g[2][0],
        g[1][1], g[1][2] + g[2][1],
        g[2][2]
    };
}

template <typename Scalar>
STROKE_DEVICES_INLINE stroke::SymmetricMat<2, Scalar> to_symmetric_gradient(const glm::mat<2, 2, Scalar>& g)
{
    return {
        g[0][0], g[0][1] + g[1][0],
        g[1][1]
    };
}

template <typename Scalar>
STROKE_DEVICES_INLINE glm::mat<3, 3, Scalar> to_mat_gradient(const stroke::SymmetricMat<3, Scalar>& g)
{
    constexpr auto half = Scalar(0.5);
    // clang-format off
    return {
        g[0],           half * g[1],    half * g[2],
        half * g[1],           g[3],    half * g[4],
        half * g[2],    half * g[4],           g[5]
    };
    // clang-format on
}

template <typename Scalar>
STROKE_DEVICES_INLINE glm::mat<2, 2, Scalar> to_mat_gradient(const stroke::SymmetricMat<2, Scalar>& g)
{
    constexpr auto half = Scalar(0.5);
    // clang-format off
    return {
        g[0],           half * g[1],
        half * g[1],           g[2]
    };
    // clang-format on
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, Scalar> det(const glm::mat<n_dims, n_dims, Scalar>& mat, Scalar grad)
{
    return cofactor(mat) * grad;
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::vec<n_dims, Scalar>, glm::vec<n_dims, Scalar>> dot(const glm::vec<n_dims, Scalar>& a, const glm::vec<n_dims, Scalar>& b, Scalar incoming_grad)
{
    return { b * incoming_grad, a * incoming_grad };
}

template <typename Scalar>
STROKE_DEVICES_INLINE TwoGrads<glm::vec<3, Scalar>, glm::vec<3, Scalar>> cross(const glm::vec<3, Scalar>& a, const glm::vec<3, Scalar>& b, const glm::vec<3, Scalar>& incoming_grad)
{
    return { cross(b, incoming_grad), cross(incoming_grad, a) };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE glm::vec<n_dims, Scalar> length(const glm::vec<n_dims, Scalar>& vec, Scalar grad)
{
    const auto l = grad / length(vec);
    return vec * l;
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE glm::vec<n_dims, Scalar> length(const glm::vec<n_dims, Scalar>& vec, Scalar grad, Scalar cached_length)
{
    const auto l = grad / cached_length;
    return vec * l;
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::vec<n_dims, Scalar>, glm::vec<n_dims, Scalar>> distance(const glm::vec<n_dims, Scalar>& a, const glm::vec<n_dims, Scalar>& b, Scalar grad)
{
    const auto vec = a - b;
    const auto l = grad / length(vec);
    return { vec * l, -vec * l };
}

template <typename Scalar>
STROKE_DEVICES_INLINE TwoGrads<glm::vec<3, Scalar>, Scalar> divide_a_by_b(const glm::vec<3, Scalar>& a, Scalar b, const glm::vec<3, Scalar>& incoming_grad)
{
    static_assert(std::is_floating_point_v<Scalar>);
    const auto a_grad = incoming_grad / b;
    //    *b_grad = sum(-incoming_grad * a / (b * b));
    const auto b_grad = sum(-a_grad * a / b); // same, but numerically more stable
    return { a_grad, b_grad };
}

template <typename Scalar, int n_dims_l, int n_dims_m, int n_dims_r>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_dims_m, n_dims_l, Scalar>, glm::mat<n_dims_r, n_dims_m, Scalar>>
matmul(const glm::mat<n_dims_m, n_dims_l, Scalar>& a, const glm::mat<n_dims_r, n_dims_m, Scalar>& b, const glm::mat<n_dims_r, n_dims_l, Scalar>& grad)
{
    return { grad * transpose(b), transpose(a) * grad };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<stroke::SymmetricMat<n_dims, Scalar>, glm::mat<n_dims, n_dims, Scalar>>
matmul(const stroke::SymmetricMat<n_dims, Scalar>& a, const glm::mat<n_dims, n_dims, Scalar>& b, const glm::mat<n_dims, n_dims, Scalar>& grad)
{
    using mat_t = glm::mat<n_dims, n_dims, Scalar>;
    return { to_symmetric_gradient(grad * transpose(b)), mat_t(a) * grad };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_dims, n_dims, Scalar>, stroke::SymmetricMat<n_dims, Scalar>>
matmul(const glm::mat<n_dims, n_dims, Scalar>& a, const stroke::SymmetricMat<n_dims, Scalar>& b, const glm::mat<n_dims, n_dims, Scalar>& grad)
{
    using mat_t = glm::mat<n_dims, n_dims, Scalar>;
    return { grad * mat_t(b), to_symmetric_gradient(transpose(a) * grad) };
}

template <typename Scalar, int n_cols, int n_rows>
STROKE_DEVICES_INLINE TwoGrads<glm::mat<n_cols, n_rows, Scalar>, glm::vec<n_cols, Scalar>>
matvecmul(const glm::mat<n_cols, n_rows, Scalar>& a, const glm::vec<n_cols, Scalar>& b, const glm::vec<n_rows, Scalar>& grad)
{
    return { outerProduct(grad, b), transpose(a) * grad };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<stroke::Cov<n_dims, Scalar>, glm::vec<n_dims, Scalar>>
matvecmul(const stroke::Cov<n_dims, Scalar>& a, const glm::vec<n_dims, Scalar>& b, const glm::vec<n_dims, Scalar>& grad)
{
    return { grad::to_symmetric_gradient(outerProduct(grad, b)), a * grad };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE TwoGrads<SymmetricMat<n_dims, Scalar>, glm::mat<n_dims, n_dims, Scalar>>
affine_transform(const SymmetricMat<n_dims, Scalar>& S, const glm::mat<n_dims, n_dims, Scalar>& M, const stroke::SymmetricMat<n_dims, Scalar>& grad)
{
    using mat_t = glm::mat<3, 3, Scalar>;
    const auto MS = M * mat_t(S);
    const auto Mt = transpose(M);
    // return MS * Mt;
    const auto [grad_MS, grad_Mt] = stroke::grad::matmul(MS, Mt, to_mat_gradient(grad));

    // const auto MS = M * mat_t(S);
    const auto [grad_M, grad_S] = stroke::grad::matmul(M, S, grad_MS);

    return { grad_S, (grad_M + transpose(grad_Mt)) };
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, Scalar> inverse_cached(const glm::mat<n_dims, n_dims, Scalar>& inverse_mat, const glm::mat<n_dims, n_dims, Scalar>& grad)
{
    // from https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf
    const auto ti = transpose(inverse_mat);
    return -ti * grad * ti;
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, Scalar> inverse(const glm::mat<n_dims, n_dims, Scalar>& mat, const glm::mat<n_dims, n_dims, Scalar>& grad)
{
    return inverse_cached(inverse(mat), grad);
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE stroke::SymmetricMat<n_dims, Scalar> inverse_cached(const stroke::SymmetricMat<n_dims, Scalar>& inverse_mat, const stroke::SymmetricMat<n_dims, Scalar>& grad)
{
    // from https://people.maths.ox.ac.uk/gilesm/files/AD2008.pdf
    // ignoring the transpose since we are symmetric
    const auto i = to_glm(inverse_mat);
    return -to_symmetric_gradient(i * to_mat_gradient(grad) * i);
}

template <typename Scalar, int n_dims>
STROKE_DEVICES_INLINE stroke::SymmetricMat<n_dims, Scalar> inverse(const stroke::SymmetricMat<n_dims, Scalar>& mat, const stroke::SymmetricMat<n_dims, Scalar>& grad)
{
    return inverse_cached(inverse(mat), grad);
}

} // namespace stroke::grad
