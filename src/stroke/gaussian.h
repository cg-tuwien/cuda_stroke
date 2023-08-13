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

#include "matrix.h"

namespace stroke::gaussian {

template <glm::length_t n_dims, typename scalar_t>
scalar_t eval_exponential_inv_C(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& inversed_covariance, const glm::vec<n_dims, scalar_t>& point)
{
    const auto t = point - centre;
    const auto v = scalar_t(-0.5) * glm::dot(t, (inversed_covariance * t));
    return stroke::exp(v);
}
template <typename scalar_t>
scalar_t eval_exponential_inv_C(const scalar_t& centre, const scalar_t& inv_variance, const scalar_t& point)
{
    const auto t = point - centre;
    const auto v = scalar_t(-0.5) * sq(t) * inv_variance;
    return stroke::exp(v);
}

template <glm::length_t n_dims, typename scalar_t>
scalar_t eval_exponential(const glm::vec<n_dims, scalar_t>& centre, const Cov<n_dims, scalar_t>& covariance, const glm::vec<n_dims, scalar_t>& point)
{
    return eval_exponential_inv_C(centre, inverse(covariance), point);
}
template <typename scalar_t>
scalar_t eval_exponential(const scalar_t& centre, const scalar_t& variance, const scalar_t& point)
{
    return eval_exponential_inv_C(centre, 1 / variance, point);
}

template <typename scalar_t,
    std::enable_if_t<std::is_floating_point<scalar_t>::value, bool> = true>
scalar_t norm_factor(const scalar_t& variance)
{
    constexpr auto factor = scalar_t(gcem::sqrt(2 * glm::pi<double>()));
    static_assert(factor > 0); // make sure factor is consteval
    return 1 / (sqrt(variance) * factor);
}

template <glm::length_t n_dims, typename scalar_t>
scalar_t norm_factor(const Cov<n_dims, scalar_t>& covariance)
{
    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));
    return 1 / sqrt(factor * det(covariance));
}

} // namespace stroke::gaussian
