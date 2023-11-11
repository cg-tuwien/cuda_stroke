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
#include <cuda/std/tuple>
#include <gcem.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/glm.hpp>

#include "stroke/cuda_compat.h"
#include "stroke/grad/linalg.h"
#include "stroke/grad/scalar_functions.h"
#include "stroke/grad/util.h"
#include "stroke/linalg.h"

namespace stroke::grad::gaussian {

namespace detail {
    template <typename scalar_t>
    STROKE_DEVICES_INLINE Cov2<scalar_t> convert_to_symmetric(const glm::mat<2, 2, scalar_t>& mat)
    {
        return { mat[0][0], mat[0][1] + mat[1][0], mat[1][1] };
    }

    template <typename scalar_t>
    STROKE_DEVICES_INLINE Cov3<scalar_t> convert_to_symmetric(const glm::mat<3, 3, scalar_t>& mat)
    {
        // clang-format off
        return { mat[0][0],         mat[0][1] + mat[1][0],          mat[0][2] + mat[2][0],
                                                mat[1][1],          mat[1][2] + mat[2][1],
                                                                                mat[2][2]};
        // clang-format on
    }
} // namespace detail

template <glm::length_t n_dims, typename scalar_t>
STROKE_DEVICES_INLINE Cov<n_dims, scalar_t> norm_factor(const Cov<n_dims, scalar_t>& covariance, scalar_t incoming_grad)
{
    constexpr auto factor = scalar_t(gcem::pow(2 * glm::pi<double>(), double(n_dims)));

    const auto d = det(covariance);
    const auto fd = factor * d;

    const auto grad_det = -incoming_grad * factor / (2 * stroke::sqrt(fd) * fd);
    const auto grad_cov = stroke::grad::det(to_glm(covariance), grad_det);
    return detail::convert_to_symmetric(grad_cov);
}

} // namespace stroke::grad::gaussian
