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

#include "stroke/grad/linalg.h"
#include "stroke/linalg.h"

TEST_CASE("stroke linalg gradients")
{
    SECTION("dot")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<glm::dvec3, glm::dvec3>(input);
            return stroke::pack_tensor<double>(dot(a, b));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<glm::dvec3, glm::dvec3>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::dot(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto test_data_host = whack::make_tensor<double>(whack::Location::Host, { 3, 2, 1, 2, 4, 3 }, 6);
        stroke::check_gradient(fun, fun_grad, test_data_host, 0.0000001);
    }

    SECTION("det")
    {
        const auto fun = [](const whack::Tensor<float, 1>& input) {
            const auto mat = stroke::extract<glm::mat3>(input);
            return stroke::pack_tensor<float>(det(mat));
        };

        const auto fun_grad = [](const whack::Tensor<float, 1>& input, const whack::Tensor<float, 1>& grad_output) {
            const auto mat = stroke::extract<glm::mat3>(input);
            const auto incoming_grad = stroke::extract<float>(grad_output);
            return stroke::pack_tensor<float>(stroke::grad::det(mat, incoming_grad));
        };

        const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, { 3, 2, 1, 2, 4, 3, 0, 1, 2 }, 9);
        stroke::check_gradient(fun, fun_grad, test_data_host);
    }
    SECTION("length")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<glm::dvec3>(input);
            return stroke::pack_tensor<double>(length(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<glm::dvec3>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const auto grad_a = stroke::grad::length(a, incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };

        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(glm::dvec3(1, 2, 3)), 0.0000001);
    }
    SECTION("length with cache")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<glm::dvec3>(input);
            return stroke::pack_tensor<double>(length(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<glm::dvec3>(input);
            const auto incoming_grad = stroke::extract<double>(grad_output);
            const auto grad_a = stroke::grad::length(a, incoming_grad, length(a));
            return stroke::pack_tensor<double>(grad_a);
        };

        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(glm::dvec3(1, 2, 3)), 0.0000001);
    }
    SECTION("matmul")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<glm::dmat3, glm::dmat3>(input);
            return stroke::pack_tensor<double>(a * b);
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<glm::dmat3, glm::dmat3>(input);
            const auto incoming_grad = stroke::extract<glm::dmat3>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::matmul(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto m1 = glm::dmat3(1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3);
        const auto m2 = glm::dmat3(11.1, 11.2, 11.3, 22.1, 22.2, 22.3, 33.1, 33.2, 33.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1, m2), 0.0000001);
    }
    SECTION("matvecmul")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<glm::dmat3, glm::dvec3>(input);
            return stroke::pack_tensor<double>(a * b);
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<glm::dmat3, glm::dvec3>(input);
            const auto incoming_grad = stroke::extract<glm::dvec3>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::matvecmul(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto m = glm::dmat3(1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3);
        const auto v = glm::dvec3(11.1, 11.2, 11.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m, v), 0.0000001);
    }

    SECTION("cov3 -> mat3")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<stroke::Cov3<double>>(input);
            return stroke::pack_tensor<double>(glm::dmat3(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<stroke::Cov3<double>>(input);
            const auto incoming_grad = stroke::extract<glm::dmat3>(grad_output);
            const auto grad_a = stroke::grad::from_mat_gradient(incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };

        const auto m1 = stroke::Cov3<double>(1.1, 1.2, 1.3, 2.2, 2.3, 3.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1), 0.0000001);
    }

    SECTION("mat3 -> cov3")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto a = stroke::extract<glm::dmat3>(input);
            return stroke::pack_tensor<double>(stroke::Cov3_d(a));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto a = stroke::extract<glm::dmat3>(input);
            const auto incoming_grad = stroke::extract<stroke::Cov3_d>(grad_output);
            const auto grad_a = stroke::grad::from_symmetric_gradient(incoming_grad);
            return stroke::pack_tensor<double>(grad_a);
        };

        const auto m1 = glm::dmat3(11.1, 11.2, 11.3, 22.1, 22.2, 22.3, 33.1, 33.2, 33.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1), 0.0000001);
    }

    SECTION("matmul with symmetric * normal")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<stroke::Cov3_d, glm::dmat3>(input);
            return stroke::pack_tensor<double>(glm::dmat3(a) * b);
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<stroke::Cov3_d, glm::dmat3>(input);
            const auto incoming_grad = stroke::extract<glm::dmat3>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::matmul(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto m1 = stroke::Cov3<double>(1.1, 1.2, 1.3, 2.2, 2.3, 3.3);
        const auto m2 = glm::dmat3(11.1, 11.2, 11.3, 22.1, 22.2, 22.3, 33.1, 33.2, 33.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1, m2), 0.0000001);
    }
    SECTION("matmul with normal * symmetric")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<glm::dmat3, stroke::Cov3_d>(input);
            return stroke::pack_tensor<double>(a * glm::dmat3(b));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<glm::dmat3, stroke::Cov3_d>(input);
            const auto incoming_grad = stroke::extract<glm::dmat3>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::matmul(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto m1 = glm::dmat3(11.1, 11.2, 11.3, 22.1, 22.2, 22.3, 33.1, 33.2, 33.3);
        const auto m2 = stroke::Cov3<double>(1.1, 1.2, 1.3, 2.2, 2.3, 3.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1, m2), 0.0000001);
    }
    SECTION("affine_transform 3d")
    {
        const auto fun = [](const whack::Tensor<double, 1>& input) {
            const auto [a, b] = stroke::extract<stroke::Cov3<double>, glm::dmat3>(input);
            return stroke::pack_tensor<double>(stroke::affine_transform(a, b));
        };

        const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
            const auto [a, b] = stroke::extract<stroke::Cov3_d, glm::dmat3>(input);
            const auto incoming_grad = stroke::extract<stroke::Cov3_d>(grad_output);
            const auto [grad_a, grad_b] = stroke::grad::affine_transform(a, b, incoming_grad);
            return stroke::pack_tensor<double>(grad_a, grad_b);
        };

        const auto m1 = stroke::Cov3<double>(1.1, 1.2, 1.3, 2.2, 2.3, 3.3);
        const auto m2 = glm::dmat3(11.1, 11.2, 11.3, 22.1, 22.2, 22.3, 33.1, 33.2, 33.3);
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(m1, m2), 0.0000001);
    }
}
