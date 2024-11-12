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
#include <nvtx3/nvToolsExt.h>
#include <stroke/linalg.h>
#include <stroke/unittest/random_entity.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <whack/Tensor.h>
#include <whack/kernel.h>
#include <whack/random/generators.h>

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

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    REQUIRE(error == cudaSuccess);
    REQUIRE(deviceCount > 0);

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
                auto R1 = glm::mat3(stroke::random_cov<3, float>(&rng));
                auto R2 = glm::mat3(stroke::random_cov<3, float>(&rng));
                auto R3 = glm::mat3(stroke::random_cov<3, float>(&rng));
                auto M = stroke::random_matrix<3, float>(&rng);
                for (int i = 0; i < n_multiplications; ++i) {
                    R1 += (M * R1 * transpose(M)) / float(n_multiplications);
                    R2 += (M * R2 * transpose(M)) / float(n_multiplications);
                    R3 += (M * R3 * transpose(M)) / float(n_multiplications);
                }
                r_v(index) = (R1 + R2 + R3) / float(n_multiplications);
            });
        nvtxRangeEnd(nvtx_range);
    };

    BENCHMARK("affine transform stroke")
    {
        const auto nvtx_range = nvtxRangeStart("affine_transform_stroke");
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
                auto R1 = stroke::random_cov<3, float>(&rng);
                auto R2 = stroke::random_cov<3, float>(&rng);
                auto R3 = stroke::random_cov<3, float>(&rng);
                auto M = stroke::random_matrix<3, float>(&rng);
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

TEST_CASE("stroke matrix benchmarks")
{
    run_matrix_benchmarks();
}
