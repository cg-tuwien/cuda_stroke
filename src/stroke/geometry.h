/*****************************************************************************
 * Stroke
 * Copyright (C) 2023 Adam Celarek
 * Copyright (c) 2015 Wenzel Jakob (for AABB to point distance)
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
#include "scalar_functions.h"

namespace stroke::geometry {

template <glm::length_t n_dims, class scalar_t>
class Aabb {
	using Vec = glm::vec<n_dims, scalar_t>;

public:
	Vec min = {};
	Vec max = {};
	[[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dims, scalar_t> size() const { return max - min; }
	[[nodiscard]] STROKE_DEVICES_INLINE bool contains(const Vec& point) const {
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
	[[nodiscard]] STROKE_DEVICES_INLINE bool contains(const scalar_t& point) const {
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
using Aabb3d = Aabb<3, double>;
using Aabb3f = Aabb<3, float>;
using Aabb3i = Aabb<3, int>;

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE bool inside(const glm::vec<n_dimensions, T>& point, const Aabb<n_dimensions, T>& box) {
	return box.contains(point);
}

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE glm::vec<n_dimensions, T> centroid(const Aabb<n_dimensions, T>& box) {
	return (box.max + box.min) * T(0.5);
}

template <glm::length_t n_dimensions, typename T>
[[nodiscard]] STROKE_DEVICES_INLINE T distance(const Aabb<n_dimensions, T>& box, const glm::vec<n_dimensions, T>& point) {
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
[[nodiscard]] STROKE_DEVICES_INLINE T largest_distance_to(const Aabb<n_dimensions, T>& box, const glm::vec<n_dimensions, T>& point) {
	T distance_squared = 0;
	for (int i = 0; i < n_dimensions; ++i) {
		if (stroke::abs(box.min[i] - point[i]) > stroke::abs(box.max[i] - point[i]))
			distance_squared += sq(box.min[i] - point[i]);
		else
			distance_squared += sq(box.max[i] - point[i]);
	}
	return stroke::sqrt(distance_squared);
}
}
