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
