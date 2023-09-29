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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"
#include "whack/random/generators.h"
#include "whack/random/state.h"

#include "stroke/matrix_functions.h"
#include "stroke/welford.h"

struct BenchmarkResults {
    float mean;
    float four_std_dev;
};

namespace {
void run_matrix_benchmarks()
{
    constexpr auto n_welfords = 10'000;
    constexpr auto n_samples = 100;
    constexpr auto n_threads_per_block = 128;
    constexpr auto n_blocks = (n_welfords + n_threads_per_block - 1) / n_threads_per_block;
    constexpr auto location = whack::Location::Device;

    auto s = whack::random::make_state(location, n_welfords);
    auto s_view = s.view();

    whack::start_parallel(
        location, n_blocks, n_threads_per_block, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            const auto index = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
            if (index >= n_welfords)
                return;

            s_view(index) = whack::random::KernelGenerator(index, 0);
        });

    auto r = whack::make_tensor<float>(location, n_welfords);
    auto r_view = r.view();

    BENCHMARK("WeightedMeanAndCov<3, float>")
    {
        whack::start_parallel(
            location, n_blocks, n_threads_per_block, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                WHACK_UNUSED(whack_blockDim);
                WHACK_UNUSED(whack_threadIdx);
                WHACK_UNUSED(whack_blockIdx);
                const auto index = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (index >= n_welfords)
                    return;

                auto& rng = s_view(index);

                stroke::welford::WeightedMeanAndCov<3, float> welford;
                for (auto i = 0; i < n_samples; ++i) {
                    const auto weight = rng.uniform() + 0.3f;
                    welford.addValue(weight, rng.normal3());
                }

                const auto mean = welford.mean();
                const auto cov = welford.cov_matrix();
                r_view(index) = sum(mean) + sum(cov);
            });
    };

    const auto r_host = r.host_copy();
    const double sum = std::accumulate(r_host.host_vector().begin(), r_host.host_vector().end(), 0.0);
    CHECK(sum / r_host.host_vector().size() == Catch::Approx(3).epsilon(0.1)); // large margin due to impossible bessel correction for reliability type weights
}

} // namespace

TEST_CASE("stroke welford benchmark") {
	run_matrix_benchmarks();
}
