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
