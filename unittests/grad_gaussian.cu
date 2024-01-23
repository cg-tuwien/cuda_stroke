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

#include <stroke/unittest/gradcheck.h>

#include <catch2/catch_test_macros.hpp>
#include <stroke/gaussian.h>
#include <stroke/grad/gaussian.h>
#include <stroke/linalg.h>
#include <stroke/unittest/random_entity.h>

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
