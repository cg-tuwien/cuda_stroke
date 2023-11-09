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

// must come first for pretty printers to work.
#include "stroke/unittest/gradcheck.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "stroke/grad/matrix_functions.h"
#include "stroke/matrix_functions.h"
#include "whack/Tensor.h"
#include "whack/TensorView.h"
#include "whack/tensor_operations.h"

TEST_CASE("stroke gradcheck")
{
    const auto identity_function = [](const auto& input) {
        return input;
    };
    const auto identity_grad_function = [](const auto&, const auto& grad_output) {
        return grad_output;
    };
    const auto sum_function = [](const auto& input) {
        auto ret_val = whack::make_host_tensor<float>(1);
        ret_val.view()(0) = input.view()(0) + input.view()(1);
        return ret_val;
    };
    const auto sum_grad_function = [](const auto& input, const auto& grad_output) {
        auto ret_val = whack::make_host_tensor<float>(2);
        ret_val.view()(0) = grad_output.view()(0);
        ret_val.view()(1) = grad_output.view()(0);
        return ret_val;
    };
    const auto mul_function = [](const auto& input) {
        auto ret_val = whack::make_host_tensor<float>(1);
        ret_val.view()(0) = input.view()(0) * input.view()(1);
        return ret_val;
    };
    const auto mul_grad_function = [](const auto& input, const auto& grad_output) {
        auto ret_val = whack::make_host_tensor<float>(2);
        ret_val.view()(0) = grad_output.view()(0) * input.view()(1);
        ret_val.view()(1) = grad_output.view()(0) * input.view()(0);
        return ret_val;
    };

    using namespace stroke;
    SECTION("api")
    {
        const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, 42);
        check_gradient(identity_function, identity_grad_function, test_data_host);
        const auto test_data_device = whack::make_tensor<float>(whack::Location::Device, 42);
        check_gradient(identity_function, identity_grad_function, test_data_device);
    }

    SECTION("analytical jacobian")
    {
        {
            auto test_data_host = whack::make_host_tensor<float>(1);
            test_data_host.view()(0) = 2;
            const auto J = gradcheck_internal::analytical_jacobian<float>(identity_function, identity_grad_function, test_data_host);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 1);
            CHECK(Jv(0, 0) == 1);
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 2;
            const auto J = gradcheck_internal::analytical_jacobian<float>(identity_function, identity_grad_function, test_data_host);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 2);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == 1);
            CHECK(Jv(0, 1) == 0);
            CHECK(Jv(1, 0) == 0);
            CHECK(Jv(1, 1) == 1);
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 2;
            const auto J = gradcheck_internal::analytical_jacobian<float>(sum_function, sum_grad_function, test_data_host);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == 1);
            CHECK(Jv(0, 1) == 1);
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 3;
            const auto J = gradcheck_internal::analytical_jacobian<float>(mul_function, mul_grad_function, test_data_host);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == 3);
            CHECK(Jv(0, 1) == 2);
        }
    }

    SECTION("numerical jacobian")
    {
        const auto dx = 0.001f;
        {
            auto test_data_host = whack::make_host_tensor<float>(1);
            test_data_host.view()(0) = 2;
            const auto J = gradcheck_internal::numerical_jacobian<float>(identity_function, test_data_host, dx);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 1);
            CHECK(Jv(0, 0) == Catch::Approx(1).epsilon(dx));
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 2;
            const auto J = gradcheck_internal::numerical_jacobian<float>(identity_function, test_data_host, dx);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 2);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == Catch::Approx(1).epsilon(dx));
            CHECK(Jv(0, 1) == Catch::Approx(0).epsilon(dx));
            CHECK(Jv(1, 0) == Catch::Approx(0).epsilon(dx));
            CHECK(Jv(1, 1) == Catch::Approx(1).epsilon(dx));
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 2;
            const auto J = gradcheck_internal::numerical_jacobian<float>(sum_function, test_data_host, dx);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == Catch::Approx(1).epsilon(dx));
            CHECK(Jv(0, 1) == Catch::Approx(1).epsilon(dx));
        }
        {
            auto test_data_host = whack::make_host_tensor<float>(2);
            test_data_host.view()(0) = 2;
            test_data_host.view()(1) = 3;
            const auto J = gradcheck_internal::numerical_jacobian<float>(mul_function, test_data_host, dx);
            const auto Jv = J.view();
            CHECK(Jv.shape().size() == 2);
            CHECK(Jv.size<0>() == 1);
            CHECK(Jv.size<1>() == 2);
            CHECK(Jv(0, 0) == Catch::Approx(3).epsilon(dx));
            CHECK(Jv(0, 1) == Catch::Approx(2).epsilon(dx));
        }
    }

    SECTION("api test with real function (host only)")
    {
        const auto fun = [](const whack::Tensor<float, 1>& input) {
            const auto mat = input.view<glm::mat3>(1)(0);
            auto ret_tensor = whack::make_tensor<float>(input.location(), 1);
            ret_tensor(0) = det(mat);
            return ret_tensor;
        };

        const auto fun_grad = [](const whack::Tensor<float, 1>& input, const whack::Tensor<float, 1>& grad_output) {
            const auto mat = input.view<glm::mat3>(1)(0);
            const auto incoming_grad = grad_output(0);

            auto grad_input = input;
            grad_input.view<glm::mat3>(1)(0) = stroke::grad::det(mat, incoming_grad);

            return grad_input;
        };

        const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, { 3, 2, 1, 2, 4, 3, 0, 1, 2 }, 9);
        check_gradient(fun, fun_grad, test_data_host);
    }

    SECTION("api test with real function (host and device)")
    {
        const auto fun = [](const whack::Tensor<float, 1>& input) {
            const auto mat_v = input.view<glm::mat3>(1);
            auto ret_tensor = whack::make_tensor<float>(input.location(), 1);
            auto ret_v = ret_tensor.view();

            whack::start_parallel(
                input.location(), 1, 1, WHACK_KERNEL(=) {
                    WHACK_UNUSED_KERNEL_PARAMS
                    ret_v(0) = det(mat_v(0));
                });

            return ret_tensor;
        };

        const auto fun_grad = [](const whack::Tensor<float, 1>& input, const whack::Tensor<float, 1>& grad_output) {
            const auto mat_v = input.view<glm::mat3>(1);
            const auto incoming_grad_v = grad_output.view();

            auto grad_input = input;
            auto grad_input_v = grad_input.view<glm::mat3>(1);

            whack::start_parallel(
                input.location(), 1, 1, WHACK_KERNEL(=) {
                    WHACK_UNUSED_KERNEL_PARAMS
                    grad_input_v(0) = stroke::grad::det(mat_v(0), incoming_grad_v(0));
                });

            return grad_input;
        };

        const auto test_data = whack::make_tensor<float>(whack::Location::Host, { 3, 2, 1, 2, 4, 3, 0, 1, 2 }, 9);
        static_assert(test_data.n_dimensions() == 1);
        check_gradient(fun, fun_grad, test_data);
        check_gradient(fun, fun_grad, test_data.device_copy());
    }
}

TEST_CASE("stroke gradcheck (test for failure)", "[!shouldfail]")
{
    const auto identity_function = [](const auto& input) {
        return input;
    };
    const auto zero_grad_function = [](const auto& input, const auto& grad_output) {
        return whack::make_tensor<typename std::remove_reference_t<decltype(input)>::value_type>(input.location(), input.dimensions());
    };

    using namespace stroke;
    // 1 input and output
    const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, 42);
    check_gradient(identity_function, zero_grad_function, test_data_host);
    const auto test_data_device = whack::make_tensor<float>(whack::Location::Device, 42);
    check_gradient(identity_function, zero_grad_function, test_data_device);
}
