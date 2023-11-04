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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "stroke/unittest/gradcheck.h"
#include "whack/Tensor.h"
#include "whack/TensorView.h"

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
        // 1 input and output
        {

            const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, 42);
            check_gradient(identity_function, identity_grad_function, test_data_host);
            const auto test_data_device = whack::make_tensor<float>(whack::Location::Device, 42);
            check_gradient(identity_function, identity_grad_function, test_data_device);
        }

        // several inputs and outputs
        {
            const auto test_data_host = whack::make_tensor<float>(whack::Location::Host, 42);
            check_gradient<float>(identity_function, identity_grad_function, { test_data_host, test_data_host, test_data_host });
            auto test_data_device = whack::make_tensor<float>(whack::Location::Device, 42);
            check_gradient<float>(identity_function, identity_grad_function, { test_data_device, test_data_device, test_data_device });
        }
    }

    SECTION("jakobian")
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
}
