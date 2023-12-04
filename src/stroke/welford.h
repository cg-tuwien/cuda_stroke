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

#include "linalg.h"

#include "cuda_compat.h"

namespace stroke::welford {

/// warning: without bessel correction. bessel is impossible according to https://stats.stackexchange.com/questions/47325/bias-correction-in-weighted-variance (see 2nd answer)
template <int N_DIMS, typename scalar_t>
struct WeightedMeanAndCov {
    using vec_t = glm::vec<N_DIMS, scalar_t>;
    //    using mat_t = glm::mat<N_DIMS, N_DIMS, scalar_t>;
    using mat_t = Cov<N_DIMS, scalar_t>;
    scalar_t w_sum = 0;
    vec_t v_mean = vec_t { 0 };
    mat_t C = mat_t { 0 };

    WeightedMeanAndCov() = default;

    STROKE_DEVICES_INLINE
    void addValue(scalar_t w, const vec_t& v)
    {
        w_sum += w;
        if (w == scalar_t(0.0))
            return;
        const auto delta1 = (v - v_mean);
        v_mean += scalar_t(w / w_sum) * delta1;
        const auto delta2 = (v - v_mean);

        C += mat_t(w * glm::outerProduct(delta1, delta2));
        assert(!isnan(C));
    }

    STROKE_DEVICES_INLINE
    scalar_t weightSum() const
    {
        return w_sum;
    }

    STROKE_DEVICES_INLINE
    vec_t mean() const
    {
        return v_mean;
    }

    STROKE_DEVICES_INLINE
    mat_t cov_matrix() const
    {
        if (w_sum == scalar_t(0.0))
            return mat_t(1);
        return C / w_sum;
    }
};

template <typename scalar_t, typename T>
struct WeightedMean {
    scalar_t w_sum = 0;
    T v_mean = T {};

    WeightedMean() = default;

    STROKE_DEVICES_INLINE
    void addValue(scalar_t w, const T& v)
    {
        w_sum += w;
        if (w == scalar_t(0.0))
            return;
        const auto delta1 = v - v_mean;
        v_mean += scalar_t(w / w_sum) * delta1; // this scalar_t cast makes the compiler happy, when compiling unit tests with autodiff.
    }

    STROKE_DEVICES_INLINE
    const T& mean() const
    {
        return v_mean;
    }
};
} // namespace stroke::welford
