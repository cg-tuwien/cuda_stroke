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

namespace {
void symmetric_matrices_work_for_shared_data()
{
    whack::start_parallel(
        whack::Location::Device, 1, 1, WHACK_DEVICE_KERNEL(=) {
            WHACK_UNUSED(whack_gridDim);
            WHACK_UNUSED(whack_blockDim);
            WHACK_UNUSED(whack_threadIdx);
            WHACK_UNUSED(whack_blockIdx);

            __shared__ stroke::Cov2<float> test_1[2];
            __shared__ stroke::Cov3<float> test_2[2];
        });
}

} // namespace

TEST_CASE("matrix symmetric works as shared data")
{
    symmetric_matrices_work_for_shared_data();
}
