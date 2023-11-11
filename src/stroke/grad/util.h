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

#include "stroke/cuda_compat.h"
#include <cassert>

namespace stroke::grad {

template <typename A1, typename A2>
struct TwoGrads {
    A1 m_left;
    A2 m_right;

    STROKE_DEVICES_INLINE
    void addTo(A1* left, A2* right)
    {
        cwise_ref_fun(&m_left, left, [](const auto& m_g, auto& g) { g += m_g; });
        cwise_ref_fun(&m_right, right, [](const auto& m_g, auto& g) { g += m_g; });
    }

    STROKE_DEVICES_INLINE
    void addTo(A1* left, bool right)
    {
        assert(right == false);
        cwise_ref_fun(&m_left, left, [](const auto& m_g, auto& g) { g += m_g; });
    }

    STROKE_DEVICES_INLINE
    void addTo(bool left, A2* right)
    {
        assert(left == false);
        cwise_ref_fun(&m_right, right, [](const auto& m_g, auto& g) { g += m_g; });
    }

    STROKE_DEVICES_INLINE
    A1 left() const
    {
        return m_left;
    }

    STROKE_DEVICES_INLINE
    A2 right() const
    {
        return m_right;
    }
};

} // namespace stroke::grad
