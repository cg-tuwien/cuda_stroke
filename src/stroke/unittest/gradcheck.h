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

#pragma once
#include <iostream>

#include <whack/pretty_printer.h>

#include <catch2/catch_test_macros.hpp>
#include <whack/Tensor.h>
#include <whack/TensorView.h>
#include <whack/kernel.h>

#include "stroke/cuda_compat.h"

namespace stroke {

template <typename scalar_t, typename Function, typename GradientFunction>
void check_gradient(Function fun, GradientFunction grad_fun, const std::vector<whack::Tensor<scalar_t, 1>>& test_input)
{
}

template <typename scalar_t, typename Function, typename GradientFunction>
void check_gradient(Function fun, GradientFunction grad_fun, const whack::Tensor<scalar_t, 1>& test_input)
{
    std::vector<whack::Tensor<scalar_t, 1>> input = { test_input };
    assert(input.size() == 1);
    check_gradient<scalar_t>(fun, grad_fun, input);
}

namespace gradcheck_internal {
    namespace detail {
        template <typename scalar_t>
        void init_index_eq_i(whack::Tensor<scalar_t, 1>* data, size_t i)
        {
            // this can't be put into analytical jacobian because:
            // error: A type local to a function ("Function" and "GradientFunction" ) cannot be used in the template argument of the enclosing parent function
            // (and any parent classes) of an extended __device__ or __host__ __device__ lambda
            const auto n = data->dimensions()[0];
            assert(i < n);
            const auto location = data->location();
            auto t = data->view();
            whack::start_parallel(
                location,
                whack::grid_dim_from_total_size(n, 256), 256, WHACK_KERNEL(=) {
                    const unsigned idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                    if (idx >= n)
                        return;
                    t(idx) = idx == i;
                });
        }

        template <typename scalar_t>
        void copy_A_to_B_at_row_i(const whack::Tensor<scalar_t, 1>& a, whack::Tensor<scalar_t, 2>* b, size_t i)
        {
            // this can't be put into analytical jacobian because:
            // error: A type local to a function ("Function" and "GradientFunction" ) cannot be used in the template argument of the enclosing parent function
            // (and any parent classes) of an extended __device__ or __host__ __device__ lambda
            assert(i < b->dimensions()[0]);
            assert(a.dimensions()[0] == b->dimensions()[1]);
            assert(a.location() == b->location());

            const auto n = a.dimensions()[0];
            auto a_v = a.view();
            auto b_v = b->view();

            whack::start_parallel(
                a.location(),
                whack::grid_dim_from_total_size(n, 256), 256, WHACK_KERNEL(=) {
                    const unsigned idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                    if (idx >= n)
                        return;
                    b_v(i, idx) = a_v(idx);
                });
        }
    } // namespace detail

    template <typename scalar_t, typename Function, typename GradientFunction>
    whack::Tensor<scalar_t, 2> analytical_jacobian(Function fun, GradientFunction grad_fun, const whack::Tensor<scalar_t, 1>& input)
    {
        const auto location = input.location();
        const auto dummy_output = fun(input);

        const auto n_inputs = input.view().size(0);
        const size_t n_outputs = dummy_output.view().size(0);

        whack::Tensor<scalar_t, 2> J = whack::make_tensor<scalar_t>(location, n_outputs, n_inputs);
        whack::Tensor<scalar_t, 1> output_grad = whack::make_tensor<scalar_t>(location, n_outputs);
        for (size_t i = 0; i < n_outputs; ++i) {
            detail::init_index_eq_i(&output_grad, i);
            // std::cout << "output_grad: " << output_grad << std::endl;
            const auto input_grad = grad_fun(input, output_grad);
            // std::cout << "input_grad: " << input_grad << std::endl;
            detail::copy_A_to_B_at_row_i(input_grad, &J, i);
        }

        // std::cout << "J: " << J << std::endl;
        return J;
    }
} // namespace gradcheck_internal

} // namespace stroke
