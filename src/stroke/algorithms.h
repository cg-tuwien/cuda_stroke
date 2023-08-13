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

#include "cuda_compat.h"

namespace stroke {

/// apply a lambda on each element
template <typename T, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE auto transform(const Array<T, N>& vec, Function fun) -> Array<decltype(fun(vec.front())), N>
{
    using ProductType = decltype(fun(vec.front()));
    Array<ProductType, N> retvec;
    for (unsigned i = 0; i < N; ++i) {
        retvec[i] = fun(vec[i]);
    }
    return retvec;
}
template <typename T, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE void transform_inplace(Array<T, N>* vec, Function fun)
{
    for (unsigned i = 0; i < N; ++i) {
        (*vec)[i] = fun((*vec)[i]);
    }
}

/// apply a lambda component wise on the elements of 2 arrays
template <typename T1, typename T2, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE auto cwise_fun(const Array<T1, N>& m1, const Array<T2, N>& m2, Function fun) -> Array<decltype(fun(m1.front(), m2.front())), N>
{
    using ProductType = decltype(fun(m1.front(), m2.front()));
    Array<ProductType, N> vec;
    for (unsigned i = 0; i < N; ++i) {
        const T1& a = m1[i];
        const T2& b = m2[i];
        vec[i] = fun(a, b);
    }
    return vec;
}
template <typename T1, typename T2, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE void cwise_inplace_fun(Array<T1, N>* m1, const Array<T2, N>& m2, Function fun)
{
    for (unsigned i = 0; i < N; ++i) {
        const T1& a = (*m1)[i];
        const T2& b = m2[i];
        (*m1)[i] = fun(a, b);
    }
}

/// apply a lambda component wise on the elements of an array and a scalar
template <typename T1, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE auto cwise_fun(const Array<T1, N>& a, const T1& b, Function fun) -> Array<decltype(fun(a.front(), b)), N>
{
    Array<decltype(fun(a.front(), b)), N> vec;
    for (unsigned i = 0; i < N; ++i) {
        vec[i] = fun(a[i], b);
    }
    return vec;
}
template <typename T1, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
STROKE_DEVICES_INLINE void cwise_inplace_fun(Array<T1, N>* m1, const T1& b, Function fun)
{
    for (unsigned i = 0; i < N; ++i) {
        const T1& a = (*m1)[i];
        (*m1)[i] = fun(a, b);
    }
}

template <typename T1, typename T2, typename SizeType, SizeType N, template <typename, SizeType> typename Array, typename Function>
__host__ __device__ STROKE_DEVICES_INLINE T2
reduce(const Array<T1, N>& m1, T2 initial, Function fun)
{
    for (unsigned i = 0; i < m1.size(); ++i) {
        const T1& a = m1[i];
        initial = fun(initial, a);
    }
    return initial;
}

} // namespace stroke
