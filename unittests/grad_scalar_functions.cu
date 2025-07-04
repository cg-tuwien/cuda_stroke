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

#include "stroke/scalar_functions.h"
#include "stroke/unittest/gradcheck.h"

#include <catch2/catch_test_macros.hpp>

#include "stroke/grad/scalar_functions.h"

TEST_CASE("stroke scalar gradients")
{
    SECTION("divide a by b")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<double, double>(input);
            return stroke::pack_tensor<double>(a / b);
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<double, double>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::divide_a_by_b(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto test_data_host = whack::make_tensor<double>(whack::Location::Host, { 3, 2 }, 2);
        stroke::check_gradient(fun, fun_grad, test_data_host, 0.0000001);
    }
    SECTION("sqrt")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<double>(input);
            return stroke::pack_tensor<double>(stroke::sqrt(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<double>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const double grad_a = stroke::grad::sqrt(a, incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };

        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(0.01), 0.0000001);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(6.0), 0.0000001);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(10.0), 0.0000001);
    }
    SECTION("erf")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<double>(input);
            return stroke::pack_tensor<double>(stroke::erf(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<double>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const double grad_a = stroke::grad::erf(a, incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };
        for (auto i = -2.; i < 2.; i += 0.2) {
            stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(i), 0.0000001);
        }
    }
    SECTION("clamp")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<double>(input);
            return stroke::pack_tensor<double>(stroke::clamp(a, 2., 8.));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<double>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const double grad_a = stroke::grad::clamp(a, 2., 8., incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };

        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(0.01), 0.0000001);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(6.0), 0.0000001);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(10.0), 0.0000001);
    }
}
