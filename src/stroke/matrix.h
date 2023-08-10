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

#include <cuda/std/array>

#include <glm/glm.hpp>

#include "scalar_functions.h"

namespace stroke {

template <unsigned n_dims, typename scalar_t>
struct SymmetricMat {
};

template <unsigned n_dims, typename scalar_t>
glm::mat<n_dims, n_dims, scalar_t> to_glm(const SymmetricMat<n_dims, scalar_t>& m)
{
    return glm::mat<n_dims, n_dims, scalar_t>(m);
}

template <unsigned n_dims, typename scalar_t>
using Cov = SymmetricMat<n_dims, scalar_t>;

template <typename scalar_t>
struct SymmetricMat<2, scalar_t> {
private:
    cuda::std::array<scalar_t, 3> m_data;

public:
    SymmetricMat(const glm::mat<2, 2, scalar_t>& mat)
        : m_data { mat[0][0], mat[0][1], mat[1][1] }
    {
    }
    SymmetricMat(scalar_t d = 0)
        : m_data { d, 0, d }
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_11)
        : m_data { m_00, m_01, m_11 }
    {
    }

    scalar_t& operator[](uint32_t i)
    {
        return m_data[i];
    }
    const scalar_t& operator[](uint32_t i) const
    {
        return m_data[i];
    }

    scalar_t& operator()(unsigned row, unsigned col)
    {
        return m_data[row + col];
    }

    const scalar_t& operator()(unsigned row, unsigned col) const
    {
        return m_data[row + col];
    }

    cuda::std::array<scalar_t, 3>& data()
    {
        return m_data;
    }

    const cuda::std::array<scalar_t, 3>& data() const
    {
        return m_data;
    }

    operator glm::mat<2, 2, scalar_t>() const
    {
        return {
            m_data[0], m_data[1],
            m_data[1], m_data[2]
        };
    }
};
static_assert(sizeof(SymmetricMat<2, float>) == 3 * 4);
static_assert(sizeof(SymmetricMat<2, double>) == 3 * 8);

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
private:
    cuda::std::array<scalar_t, 6> m_data;

public:
    SymmetricMat(const glm::mat<3, 3, scalar_t>& mat)
        : m_data { mat[0][0], mat[0][1], mat[0][2], mat[1][1], mat[1][2], mat[2][2] }
    {
    }
    SymmetricMat(scalar_t d = 0)
        : m_data { d, 0, 0, d, 0, d }
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_02, scalar_t m_11, scalar_t m_12, scalar_t m_22)
        : m_data { m_00, m_01, m_02, m_11, m_12, m_22 }
    {
    }

    scalar_t& operator[](uint32_t i)
    {
        return m_data[i];
    }
    const scalar_t& operator[](uint32_t i) const
    {
        return m_data[i];
    }

    scalar_t& operator()(unsigned row, unsigned col)
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = std::min(row, col);
        const auto max = std::max(row, col);
        if (min == 2)
            return m_data[5];
        return m_data[2 * min + max];
    }

    const scalar_t& operator()(unsigned row, unsigned col) const
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = std::min(row, col);
        const auto max = std::max(row, col);
        if (min == 2)
            return m_data[5];
        return m_data[2 * min + max];
    }

    cuda::std::array<scalar_t, 6>& data()
    {
        return m_data;
    }

    const cuda::std::array<scalar_t, 6>& data() const
    {
        return m_data;
    }

    operator glm::mat<3, 3, scalar_t>() const
    {
        return {
            m_data[0], m_data[1], m_data[2],
            m_data[1], m_data[3], m_data[4],
            m_data[2], m_data[4], m_data[5]
        };
    }
};
static_assert(sizeof(SymmetricMat<3, float>) == 6 * 4);
static_assert(sizeof(SymmetricMat<3, double>) == 6 * 8);

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
