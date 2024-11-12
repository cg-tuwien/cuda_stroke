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

// must come first for pretty printers to work.
#include "stroke/unittest/gradcheck.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "stroke/grad/linalg.h"
#include "stroke/linalg.h"
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

        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        REQUIRE(error == cudaSuccess);
        REQUIRE(deviceCount > 0);

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

    SECTION("pack tensor")
    {
        {
            const auto t = stroke::pack_tensor<float>(glm::vec2(1, 2));
            REQUIRE(t.numel() == 2);
            REQUIRE(t.n_dimensions() == 1);
            CHECK(t(0) == 1);
            CHECK(t(1) == 2);
        }

        {
            const auto t = stroke::pack_tensor<float>(glm::vec2(1, 2), glm::vec3(1, 2, 3));
            REQUIRE(t.numel() == 5);
            REQUIRE(t.n_dimensions() == 1);
            CHECK(t(0) == 1);
            CHECK(t(1) == 2);
            CHECK(t(2) == 1);
            CHECK(t(3) == 2);
            CHECK(t(4) == 3);
        }

        {
            const auto t = stroke::pack_tensor<float>(glm::vec2(1, 2), glm::vec3(1, 2, 3), 0.f);
            REQUIRE(t.numel() == 6);
            REQUIRE(t.n_dimensions() == 1);
            CHECK(t(0) == 1);
            CHECK(t(1) == 2);
            CHECK(t(2) == 1);
            CHECK(t(3) == 2);
            CHECK(t(4) == 3);
            CHECK(t(5) == 0);
        }
    }

    SECTION("extract / unpack tensor")
    {
        {
            const auto t = stroke::pack_tensor<float>(1.f, 2.f);
            CAPTURE(t);
            const auto m = stroke::extract<glm::vec2>(t);
            CHECK(m.x == 1);
            CHECK(m.y == 2);

            const auto [v1, v2] = stroke::extract<float, float>(t);
            CHECK(v1 == 1);
            CHECK(v2 == 2);
        }

        {
            const auto t = stroke::pack_tensor<float>(1.f, 2.f, 3.f, 4.f);
            CAPTURE(t);
            const auto [a1, a2] = stroke::extract<glm::vec2, glm::vec2>(t);
            CHECK(a1.x == 1);
            CHECK(a1.y == 2);
            CHECK(a2.x == 3);
            CHECK(a2.y == 4);

            const auto [b1, b2, b3] = stroke::extract<float, glm::vec2, float>(t);
            CHECK(b1 == 1);
            CHECK(b2.x == 2);
            CHECK(b2.y == 3);
            CHECK(b3 == 4);

            const auto [c1, c2, c3, c4] = stroke::extract<float, float, float, float>(t);
            CHECK(c1 == 1);
            CHECK(c2 == 2);
            CHECK(c3 == 3);
            CHECK(c4 == 4);
        }

        {
            const auto t = stroke::pack_tensor<float>(0.f, glm::vec2(1, 2), glm::vec3(3, 4, 5), glm::vec4(6, 7, 8, 9));
            CAPTURE(t);
            const auto [a1, a2, a3, a4] = stroke::extract<float, glm::vec2, glm::vec3, glm::vec4>(t);
            CHECK(a1 == 0.f);
            CHECK(a2 == glm::vec2(1, 2));
            CHECK(a3 == glm::vec3(3, 4, 5));
            CHECK(a4 == glm::vec4(6, 7, 8, 9));

            const auto [b1, b2, b3] = stroke::extract<glm::vec4, glm::vec2, glm::vec4>(t);
            CHECK(b1 == glm::vec4(0, 1, 2, 3));
            CHECK(b2 == glm::vec2(4, 5));
            CHECK(b3 == glm::vec4(6, 7, 8, 9));
        }
    }

    SECTION("api test with real function (host only)")
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
        check_gradient(fun, fun_grad, test_data_host);
    }

    SECTION("api test with real function taking several params (host only)")
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
        check_gradient(fun, fun_grad, test_data_host, 0.0000001);
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

        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        REQUIRE(error == cudaSuccess);
        REQUIRE(deviceCount > 0);
        check_gradient(fun, fun_grad, test_data.device_copy());
    }
}

TEST_CASE("stroke gradcheck (test for failure)", "[.][!shouldfail]") // hidden via [.] for now, because https://github.com/catchorg/Catch2/issues/2763
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

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    REQUIRE(error == cudaSuccess);
    REQUIRE(deviceCount > 0);
    const auto test_data_device = whack::make_tensor<float>(whack::Location::Device, 42);
    check_gradient(identity_function, zero_grad_function, test_data_device);
}
