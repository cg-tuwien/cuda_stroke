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
