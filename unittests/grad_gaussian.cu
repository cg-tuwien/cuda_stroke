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

#include "stroke/unittest/gradcheck.h"

#include <catch2/catch_test_macros.hpp>

#include "stroke/gaussian.h"
#include "stroke/grad/gaussian.h"
#include "stroke/linalg.h"
#include "test_helpers.h"

namespace {
template <uint n_dims>
void check_norm_factor()
{
    whack::random::HostGenerator<float> rnd;
    const auto fun = [](const whack::Tensor<double, 1>& input) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        return stroke::pack_tensor<double>(stroke::gaussian::norm_factor(cov));
    };

    const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        const auto incoming_grad = stroke::extract<double>(grad_output);
        const auto grad_a = stroke::grad::gaussian::norm_factor(cov, incoming_grad);
        return stroke::pack_tensor<double>(grad_a);
    };

    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(host_random_cov<n_dims, double>(&rnd)), 0.0000001);
    }
}
} // namespace

TEST_CASE("stroke gaussian gradients")
{
    SECTION("norm_factor")
    {
        check_norm_factor<2>();
        check_norm_factor<3>();
    }
}
