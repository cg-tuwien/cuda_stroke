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

#include <glm/glm.hpp>
#include <whack/random/generators.h>

#include "stroke/linalg.h"

template <glm::length_t n_dims, typename scalar_t, typename Generator>
STROKE_DEVICES_INLINE glm::mat<n_dims, n_dims, scalar_t> random_matrix(Generator* rnd)
{
    glm::mat<n_dims, n_dims, scalar_t> mat;
    for (auto c = 0; c < n_dims; ++c) {
        for (auto r = 0; r < n_dims; ++r) {
            mat[c][r] = rnd->normal();
        }
    }
    return mat;
}

template <glm::length_t n_dims, typename scalar_t, typename Generator>
STROKE_DEVICES_INLINE stroke::Cov<n_dims, scalar_t> random_cov(Generator* rnd)
{
    const auto mat = random_matrix<n_dims, scalar_t>(rnd);
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(0.05);
}

template <glm::length_t n_dims, typename scalar_t, typename Generator>
glm::mat<n_dims, n_dims, scalar_t> host_random_matrix(Generator* rnd)
{
    glm::mat<n_dims, n_dims, scalar_t> mat;
    for (auto c = 0; c < n_dims; ++c) {
        for (auto r = 0; r < n_dims; ++r) {
            mat[c][r] = rnd->normal();
        }
    }
    return mat;
}

template <glm::length_t n_dims, typename scalar_t, typename Generator>
stroke::Cov<n_dims, scalar_t> host_random_cov(Generator* rnd)
{
    const auto mat = host_random_matrix<n_dims, scalar_t>(rnd);
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(0.05);
}
