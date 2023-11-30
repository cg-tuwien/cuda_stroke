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

} // namespace stroke::grad::gaussian
