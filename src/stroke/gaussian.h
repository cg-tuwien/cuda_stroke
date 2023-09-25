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

#include <gcem.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/glm.hpp>

#include "aabb.h"
#include "cuda_compat.h"
#include "matrix.h"
#include "ray.h"

namespace stroke::gaussian {

template <glm::length_t n_dims, typename scalar_t>
struct ParamsWithWeight {
    scalar_t weight;
    glm::vec<n_dims, scalar_t> centre;
    Cov<n_dims, scalar_t> C;
};

template <typename scalar_t>
struct ParamsWithWeight<1, scalar_t> {
    scalar_t weight;
    scalar_t centre;
    scalar_t C;
};

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_exponential_inv_C(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& inversed_covariance, const glm::vec<n_dims, scalar_t>& point)
{
    const auto t = point - centre;
    const auto v = scalar_t(-0.5) * glm::dot(t, (inversed_covariance * t));
    return stroke::exp(v);
}
template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_exponential_inv_C(const scalar_t& centre, const scalar_t& inv_variance, const scalar_t& point)
{
    const auto t = point - centre;
    const auto v = scalar_t(-0.5) * sq(t) * inv_variance;
    return stroke::exp(v);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_exponential(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& covariance, const glm::vec<n_dims, scalar_t>& point)
{
    return eval_exponential_inv_C(centre, inverse(covariance), point);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_exponential(const scalar_t& centre, const scalar_t& variance, const scalar_t& point)
{
    return eval_exponential_inv_C(centre, 1 / variance, point);
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
STROKE_DEVICES_INLINE scalar_t norm_factor(const scalar_t& variance)
{
    constexpr auto factor = scalar_t(gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return 1 / (sqrt(variance) * factor);
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
STROKE_DEVICES_INLINE scalar_t norm_factor_inv_C(const scalar_t& variance)
{
    constexpr auto factor = scalar_t(1 / gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return sqrt(variance) * factor;
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t norm_factor(const Cov<n_dims, scalar_t>& covariance)
{
    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));
    return 1 / sqrt(factor * det(covariance));
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t norm_factor_inv_C(const Cov<n_dims, scalar_t>& inversed_covariance)
{
    constexpr auto factor = scalar_t(1 / gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(n_dims))));
    return factor * sqrt(det(inversed_covariance));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE ParamsWithWeight<1, scalar_t> project_on_ray_inv_C(const glm::vec<3, scalar_t>& centre, const SymmetricMat<3, scalar_t>& inversed_covariance, const Ray<3, scalar_t>& ray)
{
    // equations following the diploma thesis by Simon Fraiss (https://www.cg.tuwien.ac.at/research/publications/2022/FRAISS-2022-CGMM/)
    // little optimised
    //    const auto variance = 1 / dot(ray.direction, inversed_covariance * ray.direction);
    //    const auto position = dot(ray.direction, inversed_covariance * (centre - ray.origin)) * variance;
    //    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    // probably more optimised, no benchmark
    const auto Cxd = inversed_covariance * ray.direction;
    const auto variance = 1 / dot(ray.direction, Cxd);
    const auto position = dot(Cxd, (centre - ray.origin)) * variance;
    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    return { weight, position, variance };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE ParamsWithWeight<1, scalar_t> project_on_ray(const glm::vec<3, scalar_t>& centre, const SymmetricMat<3, scalar_t>& covariance, const Ray<3, scalar_t>& ray)
{
    return project_on_ray_inv_C(centre, inverse(covariance), ray);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t cdf_inv_C(const scalar_t& centre, const scalar_t& inv_variance, const scalar_t& x) {
	constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
	const auto scaling = inv_sqrt2 * inv_variance;
	return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_inv_C(const scalar_t& centre, const scalar_t& inv_variance, const Aabb<1, scalar_t>& box) {
	//	constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
	//	const auto scaling = inv_sqrt2 * inv_variance;
	//	const auto cdf = [&](const scalar_t& x) {
	//		return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
	//	};
	//	return cdf(box.max) - cdf(box.min);

	constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
	const auto scaling = inv_sqrt2 * inv_variance;
	return scalar_t(0.5) * (stroke::erf((box.max - centre) * scaling) - stroke::erf((box.min - centre) * scaling));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate(const scalar_t& centre, const scalar_t& variance, const Aabb<1, scalar_t>& box) {
	constexpr scalar_t sqrt2 = gcem::sqrt(scalar_t(2));
	const auto scaling = 1 / (variance * sqrt2);
	const auto cdf = [&](const scalar_t& x) {
		return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
	};
	return cdf(box.max) - cdf(box.min);
}

} // namespace stroke::gaussian
