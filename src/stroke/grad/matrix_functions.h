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

#include <glm/glm.hpp>

#include "stroke/cuda_compat.h"

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

template <typename scalar_t, int DIMS>
STROKE_DEVICES_INLINE glm::mat<DIMS, DIMS, scalar_t> det(const glm::mat<DIMS, DIMS, scalar_t>& cov, scalar_t grad)
{
    assert(glm::determinant(cov) > 0);
    return cofactor(cov) * grad;
}

} // namespace stroke::grad
