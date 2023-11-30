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
#include <vector>

#include <whack/pretty_printer.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <whack/Tensor.h>
#include <whack/TensorView.h>
#include <whack/kernel.h>

#include "stroke/cuda_compat.h"
#include "whack/tensor_operations.h"

namespace stroke {

template <typename T1, typename Tensor>
T1 extract(const Tensor& t)
{
    using scalar_t = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(scalar_t) == 0);
    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(scalar_t));
    REQUIRE(sizeof(T1) / sizeof(scalar_t) == t.numel());
    return t.template view<T1>(1)(0);
}

/// this template definition is valid, but i don't see an easy way extract from the result of split.
// template <typename T1, typename T2, typename... OtherTs, typename Tensor>
// std::tuple<T1, T2, OtherTs...> extract(const Tensor& t)
// {
//     return {};
// }

template <typename T1, typename T2, typename Tensor>
std::tuple<T1, T2> extract(const Tensor& t)
{
    using scalar_t = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T2) % sizeof(scalar_t) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(scalar_t));
    REQUIRE((sizeof(T1) + sizeof(T2)) / sizeof(scalar_t) == t.numel());
    const auto [t1, t2] = whack::split(t, sizeof(T1) / sizeof(scalar_t), sizeof(T2) / sizeof(scalar_t));
    return { extract<T1>(t1), extract<T2>(t2) };
}

template <typename T1, typename T2, typename T3, typename Tensor>
std::tuple<T1, T2, T3> extract(const Tensor& t)
{
    using scalar_t = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T2) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T3) % sizeof(scalar_t) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(T3));
    CAPTURE(sizeof(scalar_t));
    REQUIRE((sizeof(T1) + sizeof(T2) + sizeof(T3)) / sizeof(scalar_t) == t.numel());
    const auto [t1, t2, t3] = whack::split(t, sizeof(T1) / sizeof(scalar_t), sizeof(T2) / sizeof(scalar_t), sizeof(T3) / sizeof(scalar_t));
    return { extract<T1>(t1), extract<T2>(t2), extract<T3>(t3) };
}

template <typename T1, typename T2, typename T3, typename T4, typename Tensor>
std::tuple<T1, T2, T3, T4> extract(const Tensor& t)
{
    using scalar_t = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T2) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T3) % sizeof(scalar_t) == 0);
    static_assert(sizeof(T4) % sizeof(scalar_t) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(T3));
    CAPTURE(sizeof(T4));
    CAPTURE(sizeof(scalar_t));
    REQUIRE((sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4)) / sizeof(scalar_t) == t.numel());
    const auto [t1, t2, t3, t4] = whack::split(t,
        sizeof(T1) / sizeof(scalar_t),
        sizeof(T2) / sizeof(scalar_t),
        sizeof(T3) / sizeof(scalar_t),
        sizeof(T4) / sizeof(scalar_t));
    return { extract<T1>(t1), extract<T2>(t2), extract<T3>(t3), extract<T4>(t4) };
}

template <typename scalar_t, typename T1>
whack::Tensor<scalar_t, 1> pack_tensor(const T1& v)
{
    static_assert(sizeof(T1) % sizeof(scalar_t) == 0);
    auto ret_tensor = whack::make_tensor<scalar_t>(whack::Location::Host, sizeof(T1) / sizeof(scalar_t));
    ret_tensor.template view<T1>(1)(0) = v;
    return ret_tensor;
}

template <typename scalar_t, typename T1, typename... OtherTs>
whack::Tensor<scalar_t, 1> pack_tensor(const T1& v1, const OtherTs&... args)
{
    return whack::concat(pack_tensor<scalar_t>(v1), pack_tensor<scalar_t>(args)...);
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

        template <typename scalar_t>
        whack::Tensor<scalar_t, 1> copy_A_add_dx_at_i(const whack::Tensor<scalar_t, 1>& a, scalar_t dx, size_t i)
        {
            assert(i < a.numel());

            auto b = a;

            const auto n = a.numel();
            auto b_v = b.view();

            whack::start_parallel(
                a.location(), 1, 1, WHACK_KERNEL(=) {
                    WHACK_UNUSED(whack_threadIdx);
                    WHACK_UNUSED(whack_blockIdx);
                    WHACK_UNUSED(whack_blockDim);
                    WHACK_UNUSED(whack_gridDim);
                    b_v(i) = b_v(i) + dx;
                });
            return std::move(b);
        }

        template <typename scalar_t, typename IndexStoreType = unsigned int, typename IndexCalculateType = IndexStoreType>
        void set_jacobian_col_with(whack::Tensor<scalar_t, 2u>& J, size_t col, const whack::Tensor<scalar_t, 1u>& out_with_plus_dx, const whack::Tensor<scalar_t, 1u>& out_with_minus_dx, scalar_t dx)
        {
            auto Jv = J.view();
            const auto out_plus_v = out_with_plus_dx.view();
            const auto out_minus_v = out_with_minus_dx.view();
            assert(J.location() == out_with_plus_dx.location());
            assert(J.location() == out_with_minus_dx.location());
            assert(col < Jv.size(1));
            assert(out_plus_v.size(0) == Jv.size(0));
            assert(out_minus_v.size(0) == Jv.size(0));
            const auto n = Jv.size(0);

            whack::start_parallel(
                J.location(),
                whack::grid_dim_from_total_size(n, 256), 256, WHACK_KERNEL(=) {
                    const unsigned idx = whack_blockIdx.x * whack_blockDim.x + whack_threadIdx.x;
                    if (idx >= n)
                        return;
                    Jv(idx, col) = (out_plus_v(idx) - out_minus_v(idx)) / dx;
                });
        }
    } // namespace detail

    template <typename scalar_t, typename Function, typename GradientFunction>
    whack::Tensor<scalar_t, 2> analytical_jacobian(Function fun, GradientFunction grad_fun, const whack::Tensor<scalar_t, 1>& input)
    {
        const auto location = input.location();
        const auto dummy_output = fun(input);
        static_assert(dummy_output.n_dimensions() == 1);

        const auto n_inputs = input.numel();
        const size_t n_outputs = dummy_output.numel();

        whack::Tensor<scalar_t, 2> J = whack::make_tensor<scalar_t>(location, n_outputs, n_inputs);
        whack::Tensor<scalar_t, 1> output_grad = whack::make_tensor<scalar_t>(location, n_outputs);
        for (size_t i = 0; i < n_outputs; ++i) {
            detail::init_index_eq_i(&output_grad, i);
            // std::cout << "output_grad: " << output_grad << std::endl;
            const auto input_grad = grad_fun(input, output_grad);
            // std::cout << "input_grad: " << input_grad << std::endl;
            detail::copy_A_to_B_at_row_i(input_grad, &J, i);
        }

        // std::cout << "analytical_jacobian: " << J << std::endl;
        return J;
    }

    template <typename scalar_t, typename Function>
    whack::Tensor<scalar_t, 2> numerical_jacobian(Function fun, const whack::Tensor<scalar_t, 1>& input, scalar_t dx)
    {
        const auto location = input.location();
        const auto dummy_output = fun(input);
        static_assert(dummy_output.n_dimensions() == 1);

        const auto n_inputs = input.numel();
        const size_t n_outputs = dummy_output.numel();

        whack::Tensor<scalar_t, 2> J = whack::make_tensor<scalar_t>(location, n_outputs, n_inputs);
        whack::Tensor<scalar_t, 1> output_grad = whack::make_tensor<scalar_t>(location, n_outputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            const auto input_plus_dx = detail::copy_A_add_dx_at_i<scalar_t>(input, dx / 2, i);
            const auto input_minus_dx = detail::copy_A_add_dx_at_i<scalar_t>(input, -dx / 2, i);

            const auto out_with_plus_dx = fun(input_plus_dx);
            const auto out_with_minus_dx = fun(input_minus_dx);

            detail::set_jacobian_col_with<scalar_t>(J, i, out_with_plus_dx, out_with_minus_dx, dx);
        }

        // std::cout << "numerical_jacobian: " << J << std::endl;
        return J;
    }
} // namespace gradcheck_internal

template <typename scalar_t, typename Function, typename GradientFunction>
void check_gradient(Function fun, GradientFunction grad_fun, const whack::Tensor<scalar_t, 1>& test_input, scalar_t dx = 0.0001, scalar_t allowed_error_factor = 50)
{
    const auto analytical_jacobian = gradcheck_internal::analytical_jacobian<scalar_t>(fun, grad_fun, test_input).host_copy();
    const auto numerical_jacobian = gradcheck_internal::numerical_jacobian<scalar_t>(fun, test_input, dx).host_copy();
    REQUIRE(analytical_jacobian.numel() == numerical_jacobian.numel());
    REQUIRE(analytical_jacobian.n_dimensions() == 2);
    REQUIRE(analytical_jacobian.n_dimensions() == numerical_jacobian.n_dimensions());

    for (size_t output_gradient_position = 0; output_gradient_position < analytical_jacobian.dimensions()[0]; ++output_gradient_position) {
        for (size_t input_position = 0; input_position < analytical_jacobian.dimensions()[1]; ++input_position) {
            if (analytical_jacobian(output_gradient_position, input_position) != Catch::Approx(numerical_jacobian(output_gradient_position, input_position)).epsilon(dx * allowed_error_factor)) {
                // #include "stroke/unittest/gradcheck.h" must come first for pretty printers to work.
                CAPTURE(analytical_jacobian);
                CAPTURE(numerical_jacobian); // the jacobian is a copy on the host even if the computation was done on the device.
                FAIL_CHECK("Analytical and numerical jacobian do not agree. Your analytical gradient is probably incorrect!");
                return;
            }
        }
    }
}
} // namespace stroke
