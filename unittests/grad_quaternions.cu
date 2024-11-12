/****************************************************************************
 *  Copyright (C) 2024 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
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
#include <stroke/grad/quaternions.h>
#include <stroke/unittest/random_entity.h>
#include <whack/random/generators.h>

TEST_CASE("stroke quaternion toMat3 gradients")
{
    using Scalar = double;
    using Quat = glm::qua<Scalar>;
    using mat3_t = glm::mat<3, 3, Scalar>;

    const auto fun = [](const whack::Tensor<Scalar, 1>& input) {
        const auto quat = stroke::extract<Quat>(input);
        return stroke::pack_tensor<Scalar>(glm::toMat3(quat));
    };

    const auto fun_grad = [](const whack::Tensor<Scalar, 1>& input, const whack::Tensor<Scalar, 1>& grad_output) {
        const auto quat = stroke::extract<Quat>(input);
        const auto incoming_grad = stroke::extract<mat3_t>(grad_output);
        const auto grad_quat = stroke::grad::toMat3(quat, incoming_grad);
        return stroke::pack_tensor<Scalar>(grad_quat);
    };

    whack::random::HostGenerator<Scalar> rnd;
    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<Scalar>(stroke::host_random_quaternion<Scalar>(&rnd)), 0.0000001);
    }
}

TEST_CASE("stroke quaternion toMat4 gradients")
{
    using Scalar = double;
    using Quat = glm::qua<Scalar>;
    using Mat4 = glm::mat<4, 4, Scalar>;

    const auto fun = [](const whack::Tensor<Scalar, 1>& input) {
        const auto quat = stroke::extract<Quat>(input);
        return stroke::pack_tensor<Scalar>(glm::toMat4(quat));
    };

    const auto fun_grad = [](const whack::Tensor<Scalar, 1>& input, const whack::Tensor<Scalar, 1>& grad_output) {
        const auto quat = stroke::extract<Quat>(input);
        const auto incoming_grad = stroke::extract<Mat4>(grad_output);
        const auto grad_quat = stroke::grad::toMat4(quat, incoming_grad);
        return stroke::pack_tensor<Scalar>(grad_quat);
    };

    whack::random::HostGenerator<Scalar> rnd;
    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<Scalar>(stroke::host_random_quaternion<Scalar>(&rnd)), 0.0000001);
    }
}
