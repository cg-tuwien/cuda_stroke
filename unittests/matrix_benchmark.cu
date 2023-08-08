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

struct BenchmarkResults {
    float mean;
    float four_std_dev;
};

namespace {
void run_matrix_benchmarks()
{
    constexpr auto n_matrices = 1'000'000;
    constexpr auto n_multiplications = 100;
    constexpr auto n_threads_per_block = 128;
    constexpr auto n_blocks = (n_matrices + n_threads_per_block - 1) / n_threads_per_block;
    constexpr auto location = whack::Location::Device;

    auto s = whack::random::make_state(location, n_matrices);
    auto s_view = s.view();

    whack::start_parallel(
        location, n_blocks, n_threads_per_block, WHACK_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);
            const auto index = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
            if (index >= n_matrices)
                return;

            s_view(index) = whack::random::KernelGenerator(index, 0);
        });

    // ...
}

} // namespace

TEST_CASE("matrix benchmarks")
{
    run_matrix_benchmarks();
}
