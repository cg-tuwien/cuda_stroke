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

#include <whack/array.h>

#include <glm/glm.hpp>

#include "matrix_functions.h"
#include "scalar_functions.h"

namespace stroke {

template <glm::length_t n_dims, typename scalar_t>
struct SymmetricMat {
};

template <glm::length_t n_dims, typename scalar_t>
glm::mat<n_dims, n_dims, scalar_t> to_glm(const SymmetricMat<n_dims, scalar_t>& m)
{
    return glm::mat<n_dims, n_dims, scalar_t>(m);
}

template <glm::length_t n_dims, typename scalar_t>
using Cov = SymmetricMat<n_dims, scalar_t>;

namespace detail {
    constexpr glm::length_t n_elements_of_symmetric_matrix(glm::length_t n_dims)
    {
        glm::length_t n = n_dims;
        while (n_dims > 0)
            n += --n_dims;
        return n;
    }
    static_assert(n_elements_of_symmetric_matrix(2) == 3);
    static_assert(n_elements_of_symmetric_matrix(3) == 6);

    template <glm::length_t n_dims, typename scalar_t>
    struct SymmetricMatBase {
        using StorageArray = whack::Array<scalar_t, detail::n_elements_of_symmetric_matrix(n_dims)>;
        static_assert(sizeof(StorageArray) == sizeof(scalar_t) * detail::n_elements_of_symmetric_matrix(n_dims));

    protected:
        StorageArray m_data;

    public:
        SymmetricMatBase() = default;
        SymmetricMatBase(const StorageArray& d)
            : m_data(d) {};

        scalar_t& operator[](uint32_t i)
        {
            return m_data[i];
        }
        const scalar_t& operator[](uint32_t i) const
        {
            return m_data[i];
        }

        StorageArray& data()
        {
            return m_data;
        }

        const StorageArray& data() const
        {
            return m_data;
        }
    };
} // namespace detail

template <typename scalar_t>
struct SymmetricMat<2, scalar_t> : public detail::SymmetricMatBase<2, scalar_t> {
private:
    using Base = detail::SymmetricMatBase<2, scalar_t>;

public:
    using StorageArray = typename Base::StorageArray;

    /// this constructor requires an explicit typename for scalar_t, otherwise we'll generate a symmetric matrix of StorageArrays
    SymmetricMat(const StorageArray& data)
        : Base(data)
    {
    }
    explicit SymmetricMat(const glm::mat<2, 2, scalar_t>& mat)
        : Base(StorageArray({ mat[0][0], mat[0][1], mat[1][1] }))
    {
    }
    explicit SymmetricMat(scalar_t d = 0)
        : Base(StorageArray({ d, 0, d }))
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_11)
        : Base(StorageArray({ m_00, m_01, m_11 }))
    {
    }

    scalar_t& operator()(unsigned row, unsigned col)
    {
        return Base::data()[row + col];
    }

    const scalar_t& operator()(unsigned row, unsigned col) const
    {
        return Base::data()[row + col];
    }

    explicit operator glm::mat<2, 2, scalar_t>() const
    {
        return {
            Base::data()[0], Base::data()[1],
            Base::data()[1], Base::data()[2]
        };
    }
};
static_assert(sizeof(SymmetricMat<2, float>) == 3 * 4);
static_assert(sizeof(SymmetricMat<2, double>) == 3 * 8);

template <typename scalar_t>
struct Cov2 : SymmetricMat<2, scalar_t> {
private:
    using Base = SymmetricMat<2, scalar_t>;

public:
    using StorageArray = typename Base::StorageArray;

    /// this constructor requires an explicit typename for scalar_t, otherwise we'll generate a symmetric matrix of StorageArrays
    Cov2(const StorageArray& data)
        : Base(data)
    {
    }
    explicit Cov2(const glm::mat<2, 2, scalar_t>& mat)
        : Base(mat)
    {
    }
    explicit Cov2(scalar_t d = 0)
        : Base(d)
    {
    }
    Cov2(scalar_t m_00, scalar_t m_01, scalar_t m_11)
        : Base(m_00, m_01, m_11)
    {
    }
};

template <typename scalar_t>
struct SymmetricMat<3, scalar_t> : public detail::SymmetricMatBase<3, scalar_t> {
private:
    using Base = detail::SymmetricMatBase<3, scalar_t>;

public:
    using StorageArray = typename Base::StorageArray;

    /// this constructor requires an explicit typename for scalar_t, otherwise we'll generate a symmetric matrix of StorageArrays
    SymmetricMat(const StorageArray& data)
        : Base(data)
    {
    }
    explicit SymmetricMat(const glm::mat<3, 3, scalar_t>& mat)
        : Base(StorageArray({ mat[0][0], mat[0][1], mat[0][2], mat[1][1], mat[1][2], mat[2][2] }))
    {
    }
    explicit SymmetricMat(scalar_t d = 0)
        : Base(StorageArray({ d, 0, 0, d, 0, d }))
    {
    }
    SymmetricMat(scalar_t m_00, scalar_t m_01, scalar_t m_02, scalar_t m_11, scalar_t m_12, scalar_t m_22)
        : Base(StorageArray({ m_00, m_01, m_02, m_11, m_12, m_22 }))
    {
    }
    scalar_t& operator()(unsigned row, unsigned col)
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = std::min(row, col);
        const auto max = std::max(row, col);
        if (min == 2)
            return Base::data()[5];
        return Base::data()[2 * min + max];
    }

    const scalar_t& operator()(unsigned row, unsigned col) const
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = std::min(row, col);
        const auto max = std::max(row, col);
        if (min == 2)
            return Base::data()[5];
        return Base::data()[2 * min + max];
    }

    explicit operator glm::mat<3, 3, scalar_t>() const
    {
        const auto& m = Base::data();
        return {
            m[0], m[1], m[2],
            m[1], m[3], m[4],
            m[2], m[4], m[5]
        };
    }
};
static_assert(sizeof(SymmetricMat<3, float>) == 6 * 4);
static_assert(sizeof(SymmetricMat<3, double>) == 6 * 8);

template <typename scalar_t>
struct Cov3 : SymmetricMat<3, scalar_t> {
private:
    using Base = SymmetricMat<3, scalar_t>;

public:
    using StorageArray = typename Base::StorageArray;

    /// this constructor requires an explicit typename for scalar_t, otherwise we'll generate a symmetric matrix of StorageArrays
    Cov3(const StorageArray& data)
        : Base(data)
    {
    }
    explicit Cov3(const glm::mat<3, 3, scalar_t>& mat)
        : Base(mat)
    {
    }
    explicit Cov3(scalar_t d = 0)
        : Base(d)
    {
    }
    Cov3(scalar_t m_00, scalar_t m_01, scalar_t m_02, scalar_t m_11, scalar_t m_12, scalar_t m_22)
        : Base(m_00, m_01, m_02, m_11, m_12, m_22)
    {
    }
};

} // namespace stroke
