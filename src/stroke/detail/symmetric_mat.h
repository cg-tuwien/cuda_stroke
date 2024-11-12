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

#include <whack/array.h>

#include <glm/glm.hpp>

#include "stroke/cuda_compat.h"
#include "stroke/scalar_functions.h"

namespace stroke {

template <glm::length_t n_dims, typename Scalar>
struct SymmetricMat {
};

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, Scalar> to_glm(const SymmetricMat<n_dims, Scalar>& m)
{
    return glm::mat<n_dims, n_dims, Scalar>(m);
}

template <glm::length_t n_dims, typename Scalar>
using Cov = SymmetricMat<n_dims, Scalar>;

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

    template <glm::length_t n_dims, typename Scalar>
    struct SymmetricMatBase {
        using StorageArray = whack::Array<Scalar, detail::n_elements_of_symmetric_matrix(n_dims)>;
        static_assert(sizeof(StorageArray) == sizeof(Scalar) * detail::n_elements_of_symmetric_matrix(n_dims));

    protected:
        StorageArray m_data;

    public:
        SymmetricMatBase() = default;
        STROKE_DEVICES_INLINE SymmetricMatBase(const StorageArray& d)
            : m_data(d) {};

        STROKE_DEVICES_INLINE Scalar& operator[](uint32_t i)
        {
            return m_data[i];
        }
        STROKE_DEVICES_INLINE const Scalar& operator[](uint32_t i) const
        {
            return m_data[i];
        }

        STROKE_DEVICES_INLINE StorageArray& data()
        {
            return m_data;
        }

        STROKE_DEVICES_INLINE const StorageArray& data() const
        {
            return m_data;
        }
    };
} // namespace detail

template <typename Scalar>
struct SymmetricMat<2, Scalar> : public detail::SymmetricMatBase<2, Scalar> {
private:
    using Base = detail::SymmetricMatBase<2, Scalar>;

public:
    using StorageArray = typename Base::StorageArray;

    SymmetricMat() = default;

    /// this constructor requires an explicit typename for Scalar, otherwise we'll generate a symmetric matrix of StorageArrays
    STROKE_DEVICES_INLINE SymmetricMat(const StorageArray& data)
        : Base(data)
    {
    }
    STROKE_DEVICES_INLINE explicit SymmetricMat(const glm::mat<2, 2, Scalar>& mat)
        : Base(StorageArray({ mat[0][0], mat[0][1], mat[1][1] }))
    {
    }
    STROKE_DEVICES_INLINE explicit SymmetricMat(Scalar d)
        : Base(StorageArray({ d, 0, d }))
    {
    }
    STROKE_DEVICES_INLINE SymmetricMat(Scalar m_00, Scalar m_01, Scalar m_11)
        : Base(StorageArray({ m_00, m_01, m_11 }))
    {
    }
    template <typename other_Scalar>
    STROKE_DEVICES_INLINE explicit SymmetricMat(const SymmetricMat<2, other_Scalar>& other)
        : Base({ Scalar(other.data()[0]), Scalar(other.data()[1]), Scalar(other.data()[2]) })
    {
    }

    STROKE_DEVICES_INLINE Scalar& operator()(unsigned row, unsigned col)
    {
        return Base::data()[row + col];
    }

    STROKE_DEVICES_INLINE const Scalar& operator()(unsigned row, unsigned col) const
    {
        return Base::data()[row + col];
    }

    STROKE_DEVICES_INLINE explicit operator glm::mat<2, 2, Scalar>() const
    {
        return {
            Base::data()[0], Base::data()[1],
            Base::data()[1], Base::data()[2]
        };
    }
};
static_assert(sizeof(SymmetricMat<2, float>) == 3 * 4);
static_assert(sizeof(SymmetricMat<2, double>) == 3 * 8);

template <typename Scalar>
struct Cov2 : public SymmetricMat<2, Scalar> {
private:
    using Base = SymmetricMat<2, Scalar>;

public:
    using StorageArray = typename Base::StorageArray;

    Cov2() = default;

    /// this constructor requires an explicit typename for Scalar, otherwise we'll generate a symmetric matrix of StorageArrays
    STROKE_DEVICES_INLINE Cov2(const StorageArray& data)
        : Base(data)
    {
    }
    STROKE_DEVICES_INLINE explicit Cov2(const glm::mat<2, 2, Scalar>& mat)
        : Base(mat)
    {
    }
    STROKE_DEVICES_INLINE explicit Cov2(Scalar d)
        : Base(d)
    {
    }
    STROKE_DEVICES_INLINE Cov2(const Cov<2, Scalar>& other)
        : Base(other.data())
    {
    }
    STROKE_DEVICES_INLINE Cov2(Scalar m_00, Scalar m_01, Scalar m_11)
        : Base(m_00, m_01, m_11)
    {
    }
    template <typename other_Scalar>
    STROKE_DEVICES_INLINE explicit Cov2(const Cov2<other_Scalar>& other)
        : Base(Scalar(other.data()[0]), Scalar(other.data()[1]), Scalar(other.data()[2]))
    {
    }
    STROKE_DEVICES_INLINE Cov2& operator=(const Cov<2, Scalar>& other)
    {
        Base::operator=(other);
        return *this;
    }
};

template <typename Scalar>
struct SymmetricMat<3, Scalar> : public detail::SymmetricMatBase<3, Scalar> {
private:
    using Base = detail::SymmetricMatBase<3, Scalar>;

public:
    using StorageArray = typename Base::StorageArray;

    SymmetricMat() = default;

    /// this constructor requires an explicit typename for Scalar, otherwise we'll generate a symmetric matrix of StorageArrays
    STROKE_DEVICES_INLINE SymmetricMat(const StorageArray& data)
        : Base(data)
    {
    }
    // clang-format off
    STROKE_DEVICES_INLINE explicit SymmetricMat(const glm::mat<3, 3, Scalar>& mat)
        : Base(StorageArray({
            mat[0][0], Scalar(0.5) * (mat[0][1] + mat[1][0]), Scalar(0.5) * (mat[0][2] + mat[2][0]),
            mat[1][1], Scalar(0.5) * (mat[1][2] + mat[2][1]),
            mat[2][2]
            }))
    {
    }
    // clang-format on
    STROKE_DEVICES_INLINE explicit SymmetricMat(Scalar d)
        : Base(StorageArray({ d, 0, 0, d, 0, d }))
    {
    }
    STROKE_DEVICES_INLINE SymmetricMat(Scalar m_00, Scalar m_01, Scalar m_02, Scalar m_11, Scalar m_12, Scalar m_22)
        : Base(StorageArray({ m_00, m_01, m_02, m_11, m_12, m_22 }))
    {
    }
    STROKE_DEVICES_INLINE Scalar& operator()(unsigned row, unsigned col)
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = stroke::min(row, col);
        const auto max = stroke::max(row, col);
        if (min == 2)
            return Base::data()[5];
        return Base::data()[2 * min + max];
    }
    template <typename other_Scalar>
    STROKE_DEVICES_INLINE explicit SymmetricMat(const SymmetricMat<3, other_Scalar>& other)
        : Base({ Scalar(other.data()[0]), Scalar(other.data()[1]), Scalar(other.data()[2]), Scalar(other.data()[3]), Scalar(other.data()[4]), Scalar(other.data()[5]) })
    {
    }

    STROKE_DEVICES_INLINE const Scalar& operator()(unsigned row, unsigned col) const
    {
        // https://godbolt.org/z/hhr595aj5
        const auto min = std::min(row, col);
        const auto max = std::max(row, col);
        if (min == 2)
            return Base::data()[5];
        return Base::data()[2 * min + max];
    }

    STROKE_DEVICES_INLINE explicit operator glm::mat<3, 3, Scalar>() const
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

template <typename Scalar>
struct Cov3 : public SymmetricMat<3, Scalar> {
private:
    using Base = SymmetricMat<3, Scalar>;

public:
    using StorageArray = typename Base::StorageArray;

    Cov3() = default;

    /// this constructor requires an explicit typename for Scalar, otherwise we'll generate a symmetric matrix of StorageArrays
    STROKE_DEVICES_INLINE Cov3(const StorageArray& data)
        : Base(data)
    {
    }
    STROKE_DEVICES_INLINE explicit Cov3(const glm::mat<3, 3, Scalar>& mat)
        : Base(mat)
    {
    }
    STROKE_DEVICES_INLINE explicit Cov3(Scalar d)
        : Base(d)
    {
    }
    STROKE_DEVICES_INLINE Cov3(Scalar m_00, Scalar m_01, Scalar m_02, Scalar m_11, Scalar m_12, Scalar m_22)
        : Base(m_00, m_01, m_02, m_11, m_12, m_22)
    {
    }
    STROKE_DEVICES_INLINE Cov3(const Cov<3, Scalar>& other)
        : Base(other.data())
    {
    }
    template <typename other_Scalar>
    STROKE_DEVICES_INLINE explicit Cov3(const Cov<3, other_Scalar>& other)
        : Base(Scalar(other.data()[0]), Scalar(other.data()[1]), Scalar(other.data()[2]), Scalar(other.data()[3]), Scalar(other.data()[4]), Scalar(other.data()[5]))
    {
    }
    STROKE_DEVICES_INLINE Cov3& operator=(const Cov<3, Scalar>& other)
    {

        Base::operator=(other);
        return *this;
    }
};

using Cov2_f = Cov2<float>;
using Cov2_d = Cov2<double>;
using Cov3_f = Cov3<float>;
using Cov3_d = Cov3<double>;

} // namespace stroke
