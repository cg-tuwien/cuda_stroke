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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stroke/geometry.h>

using namespace stroke;
using Catch::Approx;
namespace {
template <int n_dims>
bool equals(const glm::vec<n_dims, double>& a, const glm::vec<n_dims, double>& b, double scale = 1) {
	const auto delta = glm::length(a - b);
	return delta == Approx(0).scale(scale);
}
}

TEST_CASE("geometry") {
	SECTION("aabb") {
		const auto box = geometry::Aabb<3, double> { .min = { 0.0, 1.0, 2.0 }, .max = { 10.0, 11.0, 12.0 } };
		CHECK(geometry::inside(glm::dvec3(5, 5, 5), box));
		CHECK(!geometry::inside(glm::dvec3(5, 15, 5), box));
		CHECK(equals(geometry::centroid(box), { 5.0, 6.0, 7.0 }));

		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(0, 0, 0)) == Approx(0));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(-1, 0, 0)) == Approx(1));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 0.0, 1.0, 0.0 }, .max = { 1.0, 2.0, 1.0 } }, glm::dvec3(-1, 0, 0)) == Approx(std::sqrt(2)));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 0.0, 1.0, 2.0 }, .max = { 1.0, 2.0, 1.0 } }, glm::dvec3(-1, 0, 1)) == Approx(std::sqrt(3)));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(2, 0, 0)) == Approx(1));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 10.0, 10.0, 10.0 }, .max = { 22.0, 22.0, 22.0 } }, glm::dvec3(0, 0, 0)) == Approx(std::sqrt(300)));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 10.0, 10.0, 10.0 }, .max = { 22.0, 22.0, 22.0 } }, glm::dvec3(23, 23, 23)) == Approx(std::sqrt(3)));
		CHECK(geometry::distance(geometry::Aabb<3, double> { .min = { 10.0, 10.0, 10.0 }, .max = { 22.0, 22.0, 22.0 } }, glm::dvec3(23, 22, 22)) == Approx(1));

		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(0, 0, 0)) == Approx(std::sqrt(3)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(0.5, 0.5, 0.5)) == Approx(std::sqrt(0.75)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 0.0, 0.0, 0.0 }, .max = { 1.0, 1.0, 1.0 } }, glm::dvec3(-1, 0, 0)) == Approx(std::sqrt(6)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 0.0, 1.0, 0.0 }, .max = { 1.0, 2.0, 1.0 } }, glm::dvec3(-1, 0, 0)) == Approx(std::sqrt(9)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 10.0, 10.0, 10.0 }, .max = { 22.0, 22.0, 22.0 } }, glm::dvec3(0, 0, 0)) == Approx(std::sqrt(22 * 22 * 3)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { 10.0, 10.0, 10.0 }, .max = { 22.0, 22.0, 22.0 } }, glm::dvec3(23, 23, 23)) == Approx(std::sqrt(13 * 13 * 3)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { -10.0, -10.0, -10.0 }, .max = { 20.0, 20.0, 20.0 } }, glm::dvec3(-100, 0, 0)) == Approx(std::sqrt(120 * 120 + 20 * 20 * 2)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { -10.0, -10.0, -10.0 }, .max = { 20.0, 20.0, 20.0 } }, glm::dvec3(0, -100, 0)) == Approx(std::sqrt(120 * 120 + 20 * 20 * 2)));
		CHECK(geometry::largest_distance_to(geometry::Aabb<3, double> { .min = { -10.0, -10.0, -10.0 }, .max = { 20.0, 20.0, 20.0 } }, glm::dvec3(0, 0, -100)) == Approx(std::sqrt(120 * 120 + 20 * 20 * 2)));
	}
	SECTION("aabb size") {
		CHECK(geometry::Aabb3d { { -0.5, -0.5, -0.5 }, { 0.5, 0.5, 0.5 } }.size().x == Approx(1.0));
		CHECK(geometry::Aabb3d { { -0.5, -0.5, -0.5 }, { 0.5, 0.5, 0.5 } }.size().y == Approx(1.0));
		CHECK(geometry::Aabb3d { { -0.5, -0.5, -0.5 }, { 0.5, 0.5, 0.5 } }.size().z == Approx(1.0));
		CHECK(glm::length(geometry::Aabb3d { { -0.5, -0.5, -0.5 }, { 0.5, 0.5, 0.5 } }.size()) == Approx(std::sqrt(3)));
	}
}
