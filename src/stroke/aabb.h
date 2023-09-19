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

#include "cuda_compat.h"

namespace stroke {

template <glm::length_t n_dims, class scalar_t>
class Aabb {
    using Vec = glm::vec<n_dims, scalar_t>;

public:
    Vec min = {};
    Vec max = {};
    [[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dims, scalar_t> size() const { return max - min; }
    [[nodiscard]] STROKE_DEVICES_INLINE bool contains(const Vec& point) const
    {
        return glm::all(glm::lessThanEqual(min, point)) && glm::all(glm::greaterThan(max, point));
    }
    [[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dims, scalar_t> centre() const { return (min + max) * scalar_t(0.5); }
};

template <class scalar_t>
class Aabb<1, scalar_t> {
public:
    scalar_t min = {};
    scalar_t max = {};

    [[nodiscard]] STROKE_DEVICES_INLINE scalar_t size() const { return max - min; }
    [[nodiscard]] STROKE_DEVICES_INLINE bool contains(const scalar_t& point) const
    {
        return point > min && point < max;
    }
    [[nodiscard]] STROKE_DEVICES_INLINE scalar_t centre() const { return (min + max) * scalar_t(0.5); }
};

using Aabb1d = Aabb<1, double>;
using Aabb1f = Aabb<1, float>;
using Aabb2d = Aabb<2, double>;
using Aabb2f = Aabb<2, float>;
using Aabb2i = Aabb<2, int>;
using Aabb2i64 = Aabb<2, glm::int64>;

}
