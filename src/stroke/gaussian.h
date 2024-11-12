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

template <glm::length_t n_dims, typename Scalar>
struct ParamsWithWeight {
    Scalar weight;
    glm::vec<n_dims, Scalar> centre;
    Cov<n_dims, Scalar> C;
};

template <typename Scalar>
struct ParamsWithWeight<1, Scalar> {
    Scalar weight;
    Scalar centre;
    Scalar C;
};

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_exponential_inv_C(const glm::vec<n_dims, Scalar>& centre, const Cov<n_dims, Scalar>& inversed_covariance, const glm::vec<n_dims, Scalar>& point)
{
    const auto t = point - centre;
    const auto v = Scalar(-0.5) * glm::dot(t, (inversed_covariance * t));
    return stroke::exp(v);
}
template <typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_exponential_inv_C(const Scalar& centre, const Scalar& inv_variance, const Scalar& point)
{
    const auto t = point - centre;
    const auto v = Scalar(-0.5) * sq(t) * inv_variance;
    return stroke::exp(v);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_exponential(const glm::vec<n_dims, Scalar>& centre, const Cov<n_dims, Scalar>& covariance, const glm::vec<n_dims, Scalar>& point)
{
    return eval_exponential_inv_C(centre, inverse(covariance), point);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_exponential(const Scalar& centre, const Scalar& variance, const Scalar& point)
{
    return eval_exponential_inv_C(centre, 1 / variance, point);
}

template <typename Scalar,
    std::enable_if_t<std::is_floating_point<Scalar>::value, bool> = true>
STROKE_DEVICES_INLINE Scalar integrate_exponential(const Scalar& variance)
{
    constexpr auto factor = Scalar(gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return sqrt(variance) * factor;
}

template <typename Scalar,
    std::enable_if_t<std::is_floating_point<Scalar>::value, bool> = true>
STROKE_DEVICES_INLINE Scalar norm_factor(const Scalar& variance)
{
    return 1 / integrate_exponential(variance);
}

template <typename Scalar,
    std::enable_if_t<std::is_floating_point<Scalar>::value, bool> = true>
STROKE_DEVICES_INLINE Scalar norm_factor_inv_C(const Scalar& inversed_variance)
{
    constexpr auto factor = Scalar(1 / gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return sqrt(inversed_variance) * factor;
}
template <typename Scalar>
STROKE_DEVICES_INLINE Scalar norm_factor_inv_SD(const Scalar& inversed_SD)
{
    constexpr auto factor = Scalar(1 / gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return inversed_SD * factor;
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar integrate_exponential(const Cov<n_dims, Scalar>& covariance)
{
    constexpr auto factor = Scalar(gcem::pow(2 * glm::pi<double>(), double(n_dims)));
    return sqrt(factor * stroke::max(det(covariance), Scalar(0)));
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar norm_factor(const Cov<n_dims, Scalar>& covariance)
{
    return 1 / integrate_exponential(covariance);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar norm_factor_inv_C(const Cov<n_dims, Scalar>& inversed_covariance)
{
    constexpr auto factor = Scalar(1 / gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(n_dims))));
    return factor * sqrt(det(inversed_covariance));
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_normalised(Scalar centroid, Scalar variance, Scalar point)
{
    return norm_factor(variance) * eval_exponential(centroid, variance, point);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_normalised(
    const glm::vec<n_dims, Scalar>& centroid,
    const Cov<n_dims, Scalar>& covariance,
    const glm::vec<n_dims, Scalar>& point)
{
    return norm_factor(covariance) * eval_exponential(centroid, covariance, point);
}

template <glm::length_t n_dims, typename Scalar>
STROKE_DEVICES_INLINE Scalar eval_normalised_inv_C(
    const glm::vec<n_dims, Scalar>& centroid,
    const Cov<n_dims, Scalar>& inverted_covariance,
    const glm::vec<n_dims, Scalar>& point)
{
    return norm_factor_inv_C(inverted_covariance) * eval_exponential_inv_C(centroid, inverted_covariance, point);
}

template <typename Scalar>
STROKE_DEVICES_INLINE ParamsWithWeight<1, Scalar> intersect_with_ray_inv_C(const glm::vec<3, Scalar>& centre, const SymmetricMat<3, Scalar>& inversed_covariance, const Ray<3, Scalar>& ray)
{
    // equations following the diploma thesis by Simon Fraiss (https://www.cg.tuwien.ac.at/research/publications/2022/FRAISS-2022-CGMM/)
    // little optimised
    //    const auto variance = 1 / dot(ray.direction, inversed_covariance * ray.direction);
    //    const auto position = dot(ray.direction, inversed_covariance * (centre - ray.origin)) * variance;
    //    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    // probably more optimised, no benchmark
    const auto Cxd = inversed_covariance * ray.direction;
    const auto inv_variance = dot(ray.direction, Cxd);
    if (inv_variance <= 0.001f)
        return { 0, 0, 1 };
    const auto variance = 1 / inv_variance;
    const auto position = dot(Cxd, (centre - ray.origin)) * variance;

    //    const auto weight = (1 / norm_factor_inv_C(dot_dir_cxd)) * norm_factor_inv_C(inversed_covariance) * eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);
    // const auto weight = (1 / norm_factor_inv_C(inv_variance)) * eval_normalised_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);
    const auto weight = integrate_exponential(variance) * eval_normalised_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);
    //    const auto weight = eval_exponential_inv_C(centre, inversed_covariance, ray.origin + position * ray.direction);

    return { weight, position, variance };
}

template <typename Scalar>
STROKE_DEVICES_INLINE ParamsWithWeight<1, Scalar> intersect_with_ray(const glm::vec<3, Scalar>& centre, const SymmetricMat<3, Scalar>& covariance, const Ray<3, Scalar>& ray)
{
    return intersect_with_ray_inv_C(centre, inverse(covariance), ray);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar cdf_inv_SD(const Scalar& centre, const Scalar& inv_SD, const Scalar& x)
{
    constexpr Scalar inv_sqrt2 = 1 / gcem::sqrt(Scalar(2));
    const auto scaling = inv_sqrt2 * inv_SD;
    return Scalar(0.5) * (1 + stroke::erf((x - centre) * scaling));
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar cdf_SD(const Scalar& centre, const Scalar& SD, const Scalar& x)
{
    return cdf_inv_SD(centre, 1 / SD, x);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar cdf_var(const Scalar& centre, const Scalar& var, const Scalar& x)
{
    return cdf_inv_SD(centre, 1 / stroke::sqrt(var), x);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar integrate_normalised_inv_SD(const Scalar& centre, const Scalar& inv_standard_deviation, const geometry::Aabb<1, Scalar>& box)
{
    //	constexpr Scalar inv_sqrt2 = 1 / gcem::sqrt(Scalar(2));
    //	const auto scaling = inv_sqrt2 * inv_variance;
    //	const auto cdf = [&](const Scalar& x) {
    //		return Scalar(0.5) * (1 + stroke::erf((x - centre) * scaling));
    //	};
    //	return cdf(box.max) - cdf(box.min);

    constexpr Scalar inv_sqrt2 = 1 / gcem::sqrt(Scalar(2));
    const auto scaling = inv_sqrt2 * inv_standard_deviation;
    return Scalar(0.5) * (stroke::erf((box.max - centre) * scaling) - stroke::erf((box.min - centre) * scaling));
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar integrate_normalised_SD(const Scalar& centre, const Scalar& standard_deviation, const geometry::Aabb<1, Scalar>& box)
{
    constexpr Scalar sqrt2 = gcem::sqrt(Scalar(2));
    const auto scaling = 1 / (standard_deviation * sqrt2);
    const auto cdf = [&](const Scalar& x) {
        return Scalar(0.5) * (1 + stroke::erf((x - centre) * scaling));
    };
    return cdf(box.max) - cdf(box.min);
}

template <typename Scalar>
STROKE_DEVICES_INLINE Scalar integrate_normalised_var(const Scalar& centre, const Scalar& standard_deviation, const geometry::Aabb<1, Scalar>& box)
{
    return integrate_normalised_SD(centre, stroke::sqrt(standard_deviation), box);
}

} // namespace stroke::gaussian
