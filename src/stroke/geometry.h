/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 *  Copyright (c) 2015 Wenzel Jakob (for AABB to point distance)
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

#include <glm/glm.hpp>

#include "cuda_compat.h"
#include "scalar_functions.h"

namespace stroke::geometry {

template <glm::length_t n_dims, class Scalar>
class Aabb {
    using Vec = glm::vec<n_dims, Scalar>;

public:
    Vec min = {};
    Vec max = {};
    [[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dims, Scalar> size() const { return max - min; }
    [[nodiscard]] STROKE_DEVICES_INLINE bool contains(const Vec& point) const
    {
        return glm::all(glm::lessThanEqual(min, point)) && glm::all(glm::greaterThan(max, point));
    }
    [[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dims, Scalar> centre() const { return (min + max) * Scalar(0.5); }
};

template <class Scalar>
class Aabb<1, Scalar> {
public:
    Scalar min = {};
    Scalar max = {};

    [[nodiscard]] STROKE_DEVICES_INLINE Scalar size() const { return max - min; }
    [[nodiscard]] STROKE_DEVICES_INLINE bool contains(const Scalar& point) const
    {
        return point > min && point < max;
    }
    [[nodiscard]] STROKE_DEVICES_INLINE Scalar centre() const { return (min + max) * Scalar(0.5); }
};

using Aabb1d = Aabb<1, double>;
using Aabb1f = Aabb<1, float>;
using Aabb2d = Aabb<2, double>;
using Aabb2f = Aabb<2, float>;
using Aabb2i = Aabb<2, int>;
using Aabb2i64 = Aabb<2, glm::int64>;
using Aabb3d = Aabb<3, double>;
using Aabb3f = Aabb<3, float>;
using Aabb3i = Aabb<3, int>;

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE bool inside(const glm::vec<n_dimensions, T>& point, const Aabb<n_dimensions, T>& box)
{
    return box.contains(point);
}

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dimensions, T> centroid(const Aabb<n_dimensions, T>& box)
{
    return (box.max + box.min) * T(0.5);
}

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE T distance(const Aabb<n_dimensions, T>& box, const glm::vec<n_dimensions, T>& point)
{
    T distance_squared = 0;
    for (int i = 0; i < n_dimensions; ++i) {
        T value = 0;
        if (point[i] < box.min[i])
            value = box.min[i] - point[i];
        else if (point[i] > box.max[i])
            value = point[i] - box.max[i];
        distance_squared += value * value;
    }
    return stroke::sqrt(distance_squared);
}

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE T largest_distance_to(const Aabb<n_dimensions, T>& box, const glm::vec<n_dimensions, T>& point)
{
    T distance_squared = 0;
    for (int i = 0; i < n_dimensions; ++i) {
        if (stroke::abs(box.min[i] - point[i]) > stroke::abs(box.max[i] - point[i]))
            distance_squared += sq(box.min[i] - point[i]);
        else
            distance_squared += sq(box.max[i] - point[i]);
    }
    return stroke::sqrt(distance_squared);
}
} // namespace stroke::geometry
