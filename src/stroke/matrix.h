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
#include <whack/array.h>

#include "scalar_functions.h"

namespace stroke {

template <unsigned n_dims, typename scalar_t>
struct SymmetricMat {
};

template <unsigned n_dims, typename scalar_t>
bool operator==(const SymmetricMat<n_dims, scalar_t>& a, const SymmetricMat<n_dims, scalar_t>& b)
{
    return a.data == b.data;
}

template <unsigned n_dims, typename scalar_t>
using Cov = SymmetricMat<n_dims, scalar_t>;

template <typename scalar_t>
struct SymmetricMat<2, scalar_t> {
    glm::vec<3, scalar_t> data;

    SymmetricMat(const glm::mat<2, 2, scalar_t>& mat)
        : data(mat[0][0], mat[0][1], mat[1][1])
    {
    }
    SymmetricMat(scalar_t d = 0)
        : data { d, 0, d }
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_11)
        : data(m_00, m_01, m_11)
    {
    }
};

template <typename scalar_t>
struct Cov2 : SymmetricMat<2, scalar_t> {
    Cov2(const glm::mat<2, 2, scalar_t>& mat)
        : SymmetricMat<2, scalar_t>(mat)
    {
    }
    Cov2(scalar_t d = 0)
        : SymmetricMat<2, scalar_t>(d)
    {
    }
    Cov2(scalar_t m_00, scalar_t m_01, scalar_t m_11)
        : SymmetricMat<2, scalar_t>(m_00, m_01, m_11)
    {
    }
};

template <typename scalar_t>
struct SymmetricMat<3, scalar_t> {
    whack::Array<scalar_t, 6> data;

    SymmetricMat(const glm::mat<3, 3, scalar_t>& mat)
        : data { mat[0][0], mat[0][1], mat[0][2], mat[1][1], mat[1][2], mat[2][2] }
    {
    }
    SymmetricMat(scalar_t d = 0)
        : data { d, 0, 0, d, 0, d }
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_02, scalar_t m_11, scalar_t m_12, scalar_t m_22)
        : data { m_00, m_01, m_02, m_11, m_12, m_22 }
    {
    }
};

template <typename scalar_t>
struct Cov3 : SymmetricMat<3, scalar_t> {
    Cov3(const glm::mat<3, 3, scalar_t>& mat)
        : SymmetricMat<3, scalar_t>(mat)
    {
    }
    Cov3(scalar_t d = 0)
        : SymmetricMat<3, scalar_t>(d)
    {
    }
    Cov3(scalar_t m_00, scalar_t m_01, scalar_t m_02, scalar_t m_11, scalar_t m_12, scalar_t m_22)
        : SymmetricMat<3, scalar_t>(m_00, m_01, m_02, m_11, m_12, m_22)
    {
    }
};

} // namespace stroke
