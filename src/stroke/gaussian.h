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

#include <gcem.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/glm.hpp>

#include "cuda_compat.h"
#include "geometry.h"
#include "linalg.h"
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
STROKE_DEVICES_INLINE scalar_t integrate_exponential(const scalar_t& variance)
{
    constexpr auto factor = scalar_t(gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return sqrt(variance) * factor;
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
STROKE_DEVICES_INLINE scalar_t norm_factor(const scalar_t& variance)
{
    return 1 / integrate_exponential(variance);
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
STROKE_DEVICES_INLINE scalar_t integrate_exponential(const Cov<n_dims, scalar_t>& covariance)
{
    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));
    return sqrt(factor * stroke::max(det(covariance), scalar_t(0)));
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t norm_factor(const Cov<n_dims, scalar_t>& covariance)
{
    return 1 / integrate_exponential(covariance);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t norm_factor_inv_C(const Cov<n_dims, scalar_t>& inversed_covariance)
{
    constexpr auto factor = scalar_t(1 / gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(n_dims))));
    return factor * sqrt(det(inversed_covariance));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_normalised(scalar_t centroid, scalar_t variance, scalar_t point)
{
    return norm_factor(variance) * eval_exponential(centroid, variance, point);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_normalised(
    const glm::vec<n_dims, scalar_t>& centroid,
    const Cov<n_dims, scalar_t>& covariance,
    const glm::vec<n_dims, scalar_t>& point)
{
    return norm_factor(covariance) * eval_exponential(centroid, covariance, point);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE scalar_t eval_normalised_inv_C(
    const glm::vec<n_dims, scalar_t>& centroid,
    const Cov<n_dims, scalar_t>& inverted_covariance,
    const glm::vec<n_dims, scalar_t>& point)
{
    return norm_factor_inv_C(inverted_covariance) * eval_exponential_inv_C(centroid, inverted_covariance, point);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE ParamsWithWeight<1, scalar_t> intersect_with_ray_inv_C(const glm::vec<3, scalar_t>& centre, const SymmetricMat<3, scalar_t>& inversed_covariance, const Ray<3, scalar_t>& ray)
{
    // equations following the diploma thesis by Simon Fraiss (https://www.cg.tuwien.ac.at/research/publications/2022/FRAISS-2022-CGMM/)
    // little optimised
    //    const auto variance = 1 / dot(ray.direction, inversed_covariance * ray.direction);
    //    const auto position = dot(ray.direction, inversed_covariance * (centre - ray.origin)) * variance;
    //    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    // probably more optimised, no benchmark
    const auto Cxd = inversed_covariance * ray.direction;
    const auto dot_dir_cxd = dot(ray.direction, Cxd);
    if (dot_dir_cxd <= 0.001f)
        return { 0, 0, 1 };
    const auto variance = 1 / dot_dir_cxd;
    const auto position = dot(Cxd, (centre - ray.origin)) * variance;

    //    const auto weight = (1 / norm_factor_inv_C(dot_dir_cxd)) * norm_factor_inv_C(inversed_covariance) * eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);
    const auto weight = (1 / norm_factor_inv_C(dot_dir_cxd)) * eval_normalised_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);
    //    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    return { weight, position, variance };
}

template <typename scalar_t>
STROKE_DEVICES_INLINE ParamsWithWeight<1, scalar_t> intersect_with_ray(const glm::vec<3, scalar_t>& centre, const SymmetricMat<3, scalar_t>& covariance, const Ray<3, scalar_t>& ray)
{
    return intersect_with_ray_inv_C(centre, inverse(covariance), ray);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t cdf_inv_SD(const scalar_t& centre, const scalar_t& inv_SD, const scalar_t& x)
{
    constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
    const auto scaling = inv_sqrt2 * inv_SD;
    return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t cdf_SD(const scalar_t& centre, const scalar_t& SD, const scalar_t& x)
{
    return cdf_inv_SD(centre, 1 / SD, x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t cdf_var(const scalar_t& centre, const scalar_t& var, const scalar_t& x)
{
    return cdf_inv_SD(centre, 1 / stroke::sqrt(var), x);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_normalised_inv_SD(const scalar_t& centre, const scalar_t& inv_standard_deviation, const geometry::Aabb<1, scalar_t>& box)
{
    //	constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
    //	const auto scaling = inv_sqrt2 * inv_variance;
    //	const auto cdf = [&](const scalar_t& x) {
    //		return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
    //	};
    //	return cdf(box.max) - cdf(box.min);

    constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));
    const auto scaling = inv_sqrt2 * inv_standard_deviation;
    return scalar_t(0.5) * (stroke::erf((box.max - centre) * scaling) - stroke::erf((box.min - centre) * scaling));
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_normalised_SD(const scalar_t& centre, const scalar_t& standard_deviation, const geometry::Aabb<1, scalar_t>& box)
{
    constexpr scalar_t sqrt2 = gcem::sqrt(scalar_t(2));
    const auto scaling = 1 / (standard_deviation * sqrt2);
    const auto cdf = [&](const scalar_t& x) {
        return scalar_t(0.5) * (1 + stroke::erf((x - centre) * scaling));
    };
    return cdf(box.max) - cdf(box.min);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE scalar_t integrate_normalised_var(const scalar_t& centre, const scalar_t& standard_deviation, const geometry::Aabb<1, scalar_t>& box)
{
    return integrate_normalised_SD(centre, stroke::sqrt(standard_deviation), box);
}

} // namespace stroke::gaussian
