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

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "whack/kernel.h"
#include "whack/random/generators.h"
#include "whack/random/state.h"

#include "stroke/linalg.h"
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

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    REQUIRE(error == cudaSuccess);
    REQUIRE(deviceCount > 0);

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

TEST_CASE("stroke welford benchmark")
{
    run_matrix_benchmarks();
}
