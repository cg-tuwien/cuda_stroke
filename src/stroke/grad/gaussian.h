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
#include <cuda/std/tuple>
#include <gcem.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/glm.hpp>

#include "stroke/cuda_compat.h"
#include "stroke/gaussian.h"
#include "stroke/grad/linalg.h"
#include "stroke/grad/scalar_functions.h"
#include "stroke/grad/util.h"
#include "stroke/linalg.h"

namespace stroke::grad::gaussian {

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE Cov<n_dims, scalar_t> norm_factor(const Cov<n_dims, scalar_t>& covariance, scalar_t incoming_grad)
{
    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));

    const auto d = det(covariance);
    const auto fd = factor * d;

    const auto grad_det = -incoming_grad * factor / (2 * stroke::sqrt(fd) * fd);
    const auto grad_cov = stroke::grad::det(to_glm(covariance), grad_det);
    return to_symmetric_gradient(grad_cov);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE Cov<n_dims, scalar_t> norm_factor_inv_C(const Cov<n_dims, scalar_t>& inversed_covariance, scalar_t incoming_grad)
{
    constexpr auto factor = scalar_t(1 / gcem::sqrt(gcem::pow(2 * glm::pi<double>(), double(n_dims))));
    const auto d = det(inversed_covariance);
    // return factor * sqrt_d;
    const auto grd_d = incoming_grad * factor / (2 * stroke::sqrt(d));
    const auto grad_inversed_covariance = stroke::grad::det(to_glm(inversed_covariance), grd_d);
    return to_symmetric_gradient(grad_inversed_covariance);
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
STROKE_DEVICES_INLINE scalar_t norm_factor_inv_C(const scalar_t& inversed_variance, scalar_t incoming_grad)
{
    constexpr auto factor = scalar_t(1 / gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    const auto grd_sqrt = incoming_grad * factor;
    return grd_sqrt / (2 * stroke::sqrt(inversed_variance));
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
STROKE_DEVICES_INLINE scalar_t integrate_exponential(const scalar_t& variance, scalar_t incoming_grad)
{
    constexpr auto factor = scalar_t(gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    // return sqrt(variance) * factor;
    return sqrt(variance, factor * incoming_grad);
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE Cov<n_dims, scalar_t> integrate_exponential(const Cov<n_dims, scalar_t>& covariance, scalar_t incoming_grad)
{

    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));

    const auto detcov = det(covariance);
    if (detcov <= 0)
        return {}; // handling gradient of max function

    const auto bounded_detcov = stroke::max(detcov, scalar_t(0));

    // return sqrt(factor * bounded_detcov);
    const auto grad_detcov = stroke::grad::sqrt(factor * bounded_detcov, incoming_grad) * factor;

    const auto grad_cov = stroke::grad::det(to_glm(covariance), grad_detcov);
    return to_symmetric_gradient(grad_cov);
}

template <typename scalar_t>
STROKE_DEVICES_INLINE ThreeGrads<scalar_t, scalar_t, scalar_t>
cdf_inv_SD(const scalar_t& centre, const scalar_t& inv_SD, const scalar_t& x, scalar_t incoming_grad)
{
    constexpr scalar_t inv_sqrt2 = 1 / gcem::sqrt(scalar_t(2));

    const auto scaling = inv_sqrt2 * inv_SD;
    const auto q = (x - centre) * scaling;
    // return scalar_t(0.5) * (1 + stroke::erf(q));
    const auto grd_q = stroke::grad::erf(q, incoming_grad * scalar_t(0.5));
    const auto grd_x_m_centre = grd_q * scaling;
    const auto grd_scaling = grd_q * (x - centre);

    return { -grd_x_m_centre, grd_scaling * inv_sqrt2, grd_x_m_centre };
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE ThreeGrads<glm::vec<n_dims, scalar_t>, Cov<n_dims, scalar_t>, glm::vec<n_dims, scalar_t>>
eval_exponential_inv_C(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& inversed_covariance, const glm::vec<n_dims, scalar_t>& point, scalar_t incoming_grad)
{
    const auto t = point - centre;
    const auto inv_times_t = inversed_covariance * t;
    const auto dot = glm::dot(t, inv_times_t);
    const auto v = scalar_t(-0.5) * dot;
    // return stroke::exp(v);
    const auto grad_v = stroke::exp(v) * incoming_grad;
    const auto grad_dot = grad_v * scalar_t(-0.5);
    glm::vec<n_dims, scalar_t> grad_t = {};
    glm::vec<n_dims, scalar_t> grad_inv_times_t = {};
    stroke::grad::dot(t, inv_times_t, grad_dot).addTo(&grad_t, &grad_inv_times_t);

    ThreeGrads<glm::vec<n_dims, scalar_t>, Cov<n_dims, scalar_t>, glm::vec<n_dims, scalar_t>> outgoing_grads = {};
    stroke::grad::matvecmul(inversed_covariance, t, grad_inv_times_t).addTo(&outgoing_grads.m_middle, &grad_t);
    outgoing_grads.m_left = -grad_t;
    outgoing_grads.m_right = grad_t;

    return outgoing_grads;
}

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE ThreeGrads<glm::vec<n_dims, scalar_t>, Cov<n_dims, scalar_t>, glm::vec<n_dims, scalar_t>>
eval_normalised_inv_C(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& inversed_covariance, const glm::vec<n_dims, scalar_t>& point, scalar_t incoming_grad)
{
    // return norm_factor_inv_C(inverted_covariance) * eval_exponential_inv_C(centroid, inverted_covariance, point);
    const auto norm_fct = stroke::gaussian::norm_factor_inv_C(inversed_covariance);
    const auto exp_eval = stroke::gaussian::eval_exponential_inv_C(centre, inversed_covariance, point);
    auto grad_via_eval_exp = stroke::grad::gaussian::eval_exponential_inv_C(centre, inversed_covariance, point, incoming_grad * norm_fct);
    grad_via_eval_exp.m_middle += stroke::grad::gaussian::norm_factor_inv_C(inversed_covariance, incoming_grad * exp_eval);

    return grad_via_eval_exp;
}

template <typename scalar_t>
STROKE_DEVICES_INLINE grad::ThreeGrads<glm::vec<3, scalar_t>, SymmetricMat<3, scalar_t>, Ray<3, scalar_t>>
intersect_with_ray_inv_C(const glm::vec<3, scalar_t>& centre, const SymmetricMat<3, scalar_t>& inversed_covariance, const Ray<3, scalar_t>& ray, const stroke::gaussian::ParamsWithWeight<1, scalar_t>& incoming_grad)
{
    const auto Cxd = inversed_covariance * ray.direction;
    const auto inversed_variance = dot(ray.direction, Cxd);
    grad::ThreeGrads<glm::vec<3, scalar_t>, SymmetricMat<3, scalar_t>, Ray<3, scalar_t>> grads = {};
    if (inversed_variance <= 0.001f)
        return grads;
    const auto variance = 1 / inversed_variance;
    const auto centr_m_orig = centre - ray.origin;
    const auto dot_cxd_cntr = dot(Cxd, centr_m_orig);
    const auto position = dot_cxd_cntr * variance;

    const auto t_pos = ray.origin + position * ray.direction;
    const auto exp_int = stroke::gaussian::integrate_exponential(variance);
    const auto eval_n = stroke::gaussian::eval_normalised_inv_C(centre, inversed_covariance, t_pos);
    const auto grd_exp_int = incoming_grad.weight * eval_n;
    const auto grd_eval_n = incoming_grad.weight * exp_int;
    glm::vec<3, scalar_t> grd_t_pos = {};
    stroke::grad::gaussian::eval_normalised_inv_C(centre, inversed_covariance, t_pos, grd_eval_n).addTo(&grads.m_left, &grads.m_middle, &grd_t_pos);
    grads.m_right.origin = grd_t_pos;
    grads.m_right.direction = position * grd_t_pos;
    const auto grd_position = dot(grd_t_pos, ray.direction) + incoming_grad.centre;
    const auto grd_dot_cxd_cntr = grd_position * variance;
    const auto grd_variance = grd_position * dot_cxd_cntr + incoming_grad.C + stroke::grad::gaussian::integrate_exponential(variance, grd_exp_int);

    const auto grd_inversed_variance = stroke::grad::divide_a_by_b<scalar_t>(1, inversed_variance, grd_variance).m_right;
    glm::vec<3, scalar_t> grd_Cxd = {};
    stroke::grad::dot(ray.direction, Cxd, grd_inversed_variance).addTo(&grads.m_right.direction, &grd_Cxd);
    glm::vec<3, scalar_t> grd_centr_m_orig = {};
    stroke::grad::dot(Cxd, centr_m_orig, grd_dot_cxd_cntr).addTo(&grd_Cxd, &grd_centr_m_orig);
    grads.m_left += grd_centr_m_orig;
    grads.m_right.origin -= grd_centr_m_orig;
    stroke::grad::matvecmul(inversed_covariance, ray.direction, grd_Cxd).addTo(&grads.m_middle, &grads.m_right.direction);

    return grads;
}

} // namespace stroke::grad::gaussian
