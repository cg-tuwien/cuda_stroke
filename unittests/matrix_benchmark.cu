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
#include <nvToolsExt.h>
#include <stroke/matrix.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>
#include <whack/random/generators.h>

#include "test_helpers.h"

struct BenchmarkResults {
    float mean;
    float four_std_dev;
};

namespace {
void run_matrix_benchmarks()
{
    constexpr auto n_matrices = 1'000'000;
    constexpr auto n_multiplications = 1'000;
    constexpr auto n_threads_per_block = 128;
    constexpr auto n_blocks = (n_matrices + n_threads_per_block - 1) / n_threads_per_block;
    constexpr auto location = whack::Location::Device;

    auto r = whack::make_tensor<glm::mat3>(whack::Location::Device, n_matrices);
    auto r_v = r.view();

    BENCHMARK("affine transform glm")
    {
        const auto nvtx_range = nvtxRangeStart("affine_transform_glm");
        whack::start_parallel(
            location, n_blocks, n_threads_per_block, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                WHACK_UNUSED(whack_blockDim);
                WHACK_UNUSED(whack_threadIdx);
                WHACK_UNUSED(whack_blockIdx);
                const auto index = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (index >= n_matrices)
                    return;

                auto rng = whack::random::KernelGenerator(index, 0);
                auto R1 = glm::mat3(random_cov<3, float>(&rng));
                auto R2 = glm::mat3(random_cov<3, float>(&rng));
                auto R3 = glm::mat3(random_cov<3, float>(&rng));
                auto M = random_matrix<3, float>(&rng);
                for (int i = 0; i < n_multiplications; ++i) {
                    R1 += (M * R1 * transpose(M)) / float(n_multiplications);
                    R2 += (M * R2 * transpose(M)) / float(n_multiplications);
                    R3 += (M * R3 * transpose(M)) / float(n_multiplications);
                }
                r_v(index) = (R1 + R2 + R3) / float(n_multiplications);
            });
        nvtxRangeEnd(nvtx_range);
    };

    BENCHMARK("affine transform spoke")
    {
        const auto nvtx_range = nvtxRangeStart("affine_transform_spoke");
        whack::start_parallel(
            location, n_blocks, n_threads_per_block, WHACK_KERNEL(=) {
                WHACK_UNUSED(whack_gridDim);
                WHACK_UNUSED(whack_blockDim);
                WHACK_UNUSED(whack_threadIdx);
                WHACK_UNUSED(whack_blockIdx);
                const auto index = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                if (index >= n_matrices)
                    return;

                auto rng = whack::random::KernelGenerator(index, 0);
                auto R1 = random_cov<3, float>(&rng);
                auto R2 = random_cov<3, float>(&rng);
                auto R3 = random_cov<3, float>(&rng);
                auto M = random_matrix<3, float>(&rng);
                for (int i = 0; i < n_multiplications; ++i) {
                    R1 += affine_transform(R1, M) / float(n_multiplications);
                    R2 += affine_transform(R2, M) / float(n_multiplications);
                    R3 += affine_transform(R3, M) / float(n_multiplications);
                }
                r_v(index) = glm::mat3((R1 + R2 + R3));
            });
        nvtxRangeEnd(nvtx_range);
    };

    // ...
}

} // namespace

TEST_CASE("stroke matrix benchmarks") {
	run_matrix_benchmarks();
}
