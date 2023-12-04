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

#include <type_traits>

#include "stroke/cuda_compat.h"
#include "stroke/scalar_functions.h"
#include "util.h"

namespace stroke::grad {

template <typename scalar_t>
STROKE_DEVICES_INLINE TwoGrads<scalar_t, scalar_t> divide_a_by_b(scalar_t a, scalar_t b, decltype(a / b) incoming_grad)
{
    static_assert(std::is_floating_point_v<scalar_t>);
    const auto a_grad = incoming_grad / b;
    //    *b_grad = -incoming_grad * a / (b * b);
    const auto b_grad = -a_grad * a / b; // same, but numerically more stable
    return { a_grad, b_grad };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sqrt(const scalar_t& a, scalar_t incoming_grad)
{
    static_assert(std::is_floating_point_v<scalar_t>);
    return incoming_grad / (2 * stroke::sqrt(a));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t clamp(scalar_t v, scalar_t min, scalar_t max, scalar_t incoming_grad)
{
    if (v > min && v < max)
        return incoming_grad;
    return 0;
}

} // namespace stroke::grad
