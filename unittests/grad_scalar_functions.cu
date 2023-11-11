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
}
