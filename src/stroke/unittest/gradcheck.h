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

#pragma once

#include <whack/pretty_printer.h>
// whack pretty printer must go before stroke
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <stroke/cuda_compat.h>
#include <stroke/pretty_printers.h>
#include <vector>
#include <whack/Tensor.h>
#include <whack/TensorView.h>
#include <whack/array.h>
#include <whack/kernel.h>
#include <whack/tensor_operations.h>

namespace stroke {

template <typename T1, typename Tensor>
T1 extract(const Tensor& t)
{
    using Scalar = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(Scalar));
    REQUIRE(sizeof(T1) / sizeof(Scalar) == t.numel());
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
    using Scalar = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    static_assert(sizeof(T2) % sizeof(Scalar) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(Scalar));
    REQUIRE((sizeof(T1) + sizeof(T2)) / sizeof(Scalar) == t.numel());
    const auto [t1, t2] = whack::split(t, sizeof(T1) / sizeof(Scalar), sizeof(T2) / sizeof(Scalar));
    return { extract<T1>(t1), extract<T2>(t2) };
}

template <typename T1, typename T2, typename T3, typename Tensor>
std::tuple<T1, T2, T3> extract(const Tensor& t)
{
    using Scalar = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    static_assert(sizeof(T2) % sizeof(Scalar) == 0);
    static_assert(sizeof(T3) % sizeof(Scalar) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(T3));
    CAPTURE(sizeof(Scalar));
    REQUIRE((sizeof(T1) + sizeof(T2) + sizeof(T3)) / sizeof(Scalar) == t.numel());
    const auto [t1, t2, t3] = whack::split(t, sizeof(T1) / sizeof(Scalar), sizeof(T2) / sizeof(Scalar), sizeof(T3) / sizeof(Scalar));
    return { extract<T1>(t1), extract<T2>(t2), extract<T3>(t3) };
}

template <typename T1, typename T2, typename T3, typename T4, typename Tensor>
std::tuple<T1, T2, T3, T4> extract(const Tensor& t)
{
    using Scalar = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    static_assert(sizeof(T2) % sizeof(Scalar) == 0);
    static_assert(sizeof(T3) % sizeof(Scalar) == 0);
    static_assert(sizeof(T4) % sizeof(Scalar) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(T3));
    CAPTURE(sizeof(T4));
    CAPTURE(sizeof(Scalar));
    REQUIRE((sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4)) / sizeof(Scalar) == t.numel());
    const auto [t1, t2, t3, t4] = whack::split(t,
        sizeof(T1) / sizeof(Scalar),
        sizeof(T2) / sizeof(Scalar),
        sizeof(T3) / sizeof(Scalar),
        sizeof(T4) / sizeof(Scalar));
    return { extract<T1>(t1), extract<T2>(t2), extract<T3>(t3), extract<T4>(t4) };
}


template <typename T1, typename T2, typename T3, typename T4, typename T5, typename Tensor>
std::tuple<T1, T2, T3, T4, T5> extract(const Tensor& t)
{
    using Scalar = typename Tensor::value_type;
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    static_assert(sizeof(T2) % sizeof(Scalar) == 0);
    static_assert(sizeof(T3) % sizeof(Scalar) == 0);
    static_assert(sizeof(T4) % sizeof(Scalar) == 0);
    static_assert(sizeof(T5) % sizeof(Scalar) == 0);

    REQUIRE(t.location() == whack::Location::Host);
    CAPTURE(sizeof(T1));
    CAPTURE(sizeof(T2));
    CAPTURE(sizeof(T3));
    CAPTURE(sizeof(T4));
    CAPTURE(sizeof(T5));
    CAPTURE(sizeof(Scalar));
    REQUIRE((sizeof(T1) + sizeof(T2) + sizeof(T3) + sizeof(T4) + sizeof(T5)) / sizeof(Scalar) == t.numel());
    const auto [t1, t2, t3, t4, t5] = whack::split(t,
        sizeof(T1) / sizeof(Scalar),
        sizeof(T2) / sizeof(Scalar),
        sizeof(T3) / sizeof(Scalar),
        sizeof(T4) / sizeof(Scalar),
        sizeof(T5) / sizeof(Scalar));
    return { extract<T1>(t1), extract<T2>(t2), extract<T3>(t3), extract<T4>(t4), extract<T5>(t5) };
}

template <typename Scalar, typename T1>
whack::Tensor<Scalar, 1> pack_tensor(const T1& v)
{
    static_assert(sizeof(T1) % sizeof(Scalar) == 0);
    auto ret_tensor = whack::make_tensor<Scalar>(whack::Location::Host, sizeof(T1) / sizeof(Scalar));
    ret_tensor.template view<T1>(1)(0) = v;
    return ret_tensor;
}

template <typename Scalar, typename T1, typename... OtherTs>
whack::Tensor<Scalar, 1> pack_tensor(const T1& v1, const OtherTs&... args)
{
    return whack::concat(pack_tensor<Scalar>(v1), pack_tensor<Scalar>(args)...);
}

namespace gradcheck_internal {
    namespace detail {
        template <typename Scalar>
        void init_index_eq_i(whack::Tensor<Scalar, 1>* data, size_t i)
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

        template <typename Scalar>
        void copy_A_to_B_at_row_i(const whack::Tensor<Scalar, 1>& a, whack::Tensor<Scalar, 2>* b, size_t i)
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

        template <typename Scalar>
        whack::Tensor<Scalar, 1> copy_A_add_dx_at_i(const whack::Tensor<Scalar, 1>& a, Scalar dx, size_t i)
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

        template <typename Scalar, typename IndexStoreType = unsigned int, typename IndexCalculateType = IndexStoreType>
        void set_jacobian_col_with(whack::Tensor<Scalar, 2u>& J, size_t col, const whack::Tensor<Scalar, 1u>& out_with_plus_dx, const whack::Tensor<Scalar, 1u>& out_with_minus_dx, Scalar dx)
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

    template <typename Scalar, typename Function, typename GradientFunction>
    whack::Tensor<Scalar, 2> analytical_jacobian(Function fun, GradientFunction grad_fun, const whack::Tensor<Scalar, 1>& input)
    {
        const auto location = input.location();
        const auto dummy_output = fun(input);
        static_assert(dummy_output.n_dimensions() == 1);

        const auto n_inputs = input.numel();
        const size_t n_outputs = dummy_output.numel();

        whack::Tensor<Scalar, 2> J = whack::make_tensor<Scalar>(location, n_outputs, n_inputs);
        whack::Tensor<Scalar, 1> output_grad = whack::make_tensor<Scalar>(location, n_outputs);
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

    template <typename Scalar, typename Function>
    whack::Tensor<Scalar, 2> numerical_jacobian(Function fun, const whack::Tensor<Scalar, 1>& input, Scalar dx)
    {
        const auto location = input.location();
        const auto dummy_output = fun(input);
        static_assert(dummy_output.n_dimensions() == 1);

        const auto n_inputs = input.numel();
        const size_t n_outputs = dummy_output.numel();

        whack::Tensor<Scalar, 2> J = whack::make_tensor<Scalar>(location, n_outputs, n_inputs);
        whack::Tensor<Scalar, 1> output_grad = whack::make_tensor<Scalar>(location, n_outputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            const auto input_plus_dx = detail::copy_A_add_dx_at_i<Scalar>(input, dx / 2, i);
            const auto input_minus_dx = detail::copy_A_add_dx_at_i<Scalar>(input, -dx / 2, i);

            const auto out_with_plus_dx = fun(input_plus_dx);
            const auto out_with_minus_dx = fun(input_minus_dx);

            detail::set_jacobian_col_with<Scalar>(J, i, out_with_plus_dx, out_with_minus_dx, dx);
        }

        // std::cout << "numerical_jacobian: " << J << std::endl;
        return J;
    }
} // namespace gradcheck_internal

template <typename Scalar, typename Function, typename GradientFunction>
void check_gradient(Function fun, GradientFunction grad_fun, const whack::Tensor<Scalar, 1>& test_input, Scalar dx = 0.0001, Scalar allowed_error_factor = 50)
{
    const auto analytical_jacobian = gradcheck_internal::analytical_jacobian<Scalar>(fun, grad_fun, test_input).host_copy();
    const auto numerical_jacobian = gradcheck_internal::numerical_jacobian<Scalar>(fun, test_input, dx).host_copy();
    REQUIRE(analytical_jacobian.numel() == numerical_jacobian.numel());
    REQUIRE(analytical_jacobian.n_dimensions() == 2);
    REQUIRE(analytical_jacobian.n_dimensions() == numerical_jacobian.n_dimensions());

    for (size_t output_gradient_position = 0; output_gradient_position < analytical_jacobian.dimensions()[0]; ++output_gradient_position) {
        for (size_t input_position = 0; input_position < analytical_jacobian.dimensions()[1]; ++input_position) {
            if (analytical_jacobian(output_gradient_position, input_position) != Catch::Approx(numerical_jacobian(output_gradient_position, input_position)).epsilon(dx * allowed_error_factor).margin(dx * allowed_error_factor)) {
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
