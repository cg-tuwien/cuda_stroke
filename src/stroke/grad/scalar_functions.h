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

#include <type_traits>

#include "stroke/cuda_compat.h"
#include "stroke/scalar_functions.h"
#include "util.h"

namespace stroke::grad {

template <typename scalar_t>
STROKE_DEVICES_INLINE TwoGrads<scalar_t, scalar_t> divide_a_by_b(const scalar_t& a, const scalar_t& b, const decltype(a / b)& incoming_grad)
{
    static_assert(std::is_floating_point_v<scalar_t>);
    const auto a_grad = incoming_grad / b;
    //    *b_grad = -incoming_grad * a / (b * b);
    const auto b_grad = -a_grad * a / b; // same, but numerically more stable
    return { a_grad, b_grad };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t sqrt(const scalar_t& a, const scalar_t& incoming_grad)
{
    static_assert(std::is_floating_point_v<scalar_t>);
    return incoming_grad / (2 * stroke::sqrt(a));
}
} // namespace stroke::grad
