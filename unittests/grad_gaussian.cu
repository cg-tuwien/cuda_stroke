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

#include <stroke/unittest/gradcheck.h>

#include <catch2/catch_test_macros.hpp>
#include <stroke/gaussian.h>
#include <stroke/grad/gaussian.h>
#include <stroke/linalg.h>
#include <stroke/unittest/random_entity.h>

namespace {
template <uint n_dims>
void check_norm_factor()
{
    whack::random::HostGenerator<float> rnd;
    const auto fun = [](const whack::Tensor<double, 1>& input) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        return stroke::pack_tensor<double>(stroke::gaussian::norm_factor(cov));
    };

    const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        const auto incoming_grad = stroke::extract<double>(grad_output);
        const auto grad_a = stroke::grad::gaussian::norm_factor(cov, incoming_grad);
        return stroke::pack_tensor<double>(grad_a);
    };

    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(stroke::host_random_cov<n_dims, double>(&rnd)), 0.0000001);
    }
}
template <uint n_dims>
void check_norm_factor_inv_C()
{
    whack::random::HostGenerator<float> rnd;
    const auto fun = [](const whack::Tensor<double, 1>& input) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        return stroke::pack_tensor<double>(stroke::gaussian::norm_factor_inv_C(cov));
    };

    const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        const auto incoming_grad = stroke::extract<double>(grad_output);
        const auto grad_a = stroke::grad::gaussian::norm_factor_inv_C(cov, incoming_grad);
        return stroke::pack_tensor<double>(grad_a);
    };

    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(inverse(stroke::host_random_cov<n_dims, double>(&rnd))), 0.0000001);
    }
}

template <uint n_dims>
void check_integrate_exponential()
{
    whack::random::HostGenerator<float> rnd;
    const auto fun = [](const whack::Tensor<double, 1>& input) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        return stroke::pack_tensor<double>(stroke::gaussian::integrate_exponential(cov));
    };

    const auto fun_grad = [](const whack::Tensor<double, 1>& input, const whack::Tensor<double, 1>& grad_output) {
        const auto cov = stroke::extract<stroke::Cov<n_dims, double>>(input);
        const auto incoming_grad = stroke::extract<double>(grad_output);
        const auto grad_a = stroke::grad::gaussian::integrate_exponential(cov, incoming_grad);
        return stroke::pack_tensor<double>(grad_a);
    };

    for (int i = 0; i < 10; ++i) {
        stroke::check_gradient(fun, fun_grad, stroke::pack_tensor<double>(stroke::host_random_cov<n_dims, double>(&rnd)), 0.0000001);
    }
}

template <uint n_dims>
void check_eval_exponential_normalised_inv_C()
{
    using scalar_t = double;
    using vec3_t = glm::vec<3, scalar_t>;
    using cov3_t = stroke::Cov3<scalar_t>;
    using Gaussian1d = stroke::gaussian::ParamsWithWeight<1, scalar_t>;

    whack::random::HostGenerator<scalar_t> rnd;

    for (int i = 0; i < 10; ++i) {
        const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
            const auto [g_pos, g_cov, e_pos] = stroke::extract<vec3_t, cov3_t, vec3_t>(input);
            const auto value1 = stroke::gaussian::eval_exponential_inv_C(g_pos, g_cov, e_pos);
            const auto value2 = stroke::gaussian::eval_normalised_inv_C(g_pos, g_cov, e_pos);
            return stroke::pack_tensor<scalar_t>(value1, value2);
        };

        const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
            const auto [g_pos, g_cov, e_pos] = stroke::extract<vec3_t, cov3_t, vec3_t>(input);
            const auto [grad_value1, grad_value2] = stroke::extract<scalar_t, scalar_t>(grad_output);

            auto grad_outgoing = stroke::grad::gaussian::eval_exponential_inv_C(g_pos, g_cov, e_pos, grad_value1);
            stroke::grad::gaussian::eval_normalised_inv_C(g_pos, g_cov, e_pos, grad_value2).addTo(&grad_outgoing.m_left, &grad_outgoing.m_middle, &grad_outgoing.m_right);

            return stroke::pack_tensor<scalar_t>(grad_outgoing);
        };

        const auto g_pos = rnd.normal3() * 10.;
        const auto g_cov = inverse(stroke::host_random_cov<3, double>(&rnd));
        const auto e_pos = g_pos + rnd.normal3() * 2.5;
        const auto test_data = stroke::pack_tensor<scalar_t>(g_pos, g_cov, e_pos);
        stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
    }
}
} // namespace

TEST_CASE("stroke gaussian gradients")
{
    SECTION("norm_factor")
    {
        check_norm_factor<2>();
        check_norm_factor<3>();
        check_norm_factor_inv_C<2>();
        check_norm_factor_inv_C<3>();
    }
    SECTION("integrate_exponential")
    {
        check_integrate_exponential<2>();
        check_integrate_exponential<3>();
    }

    SECTION("eval_exponential/normalised_inv_C")
    {
        check_eval_exponential_normalised_inv_C<2>();
        check_eval_exponential_normalised_inv_C<3>();
    }

    SECTION("1d norm_factor_inv_C")
    {
        using scalar_t = double;
        whack::random::HostGenerator<scalar_t> rnd;

        for (int i = 0; i < 10; ++i) {
            const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
                const auto inv_var = stroke::extract<scalar_t>(input);
                const auto norm_fct = stroke::gaussian::norm_factor_inv_C<scalar_t>(inv_var);
                return stroke::pack_tensor<scalar_t>(norm_fct);
            };

            const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
                const auto inv_var = stroke::extract<scalar_t>(input);
                const auto grad_incoming = stroke::extract<scalar_t>(grad_output);

                const auto grad_outgoing = stroke::grad::gaussian::norm_factor_inv_C<scalar_t>(inv_var, grad_incoming);

                return stroke::pack_tensor<scalar_t>(grad_outgoing);
            };

            const auto test_data = stroke::pack_tensor<scalar_t>(rnd.uniform());
            stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
        }
    }

    SECTION("intersect_with_ray_inv_C")
    {
        using scalar_t = double;
        using vec3_t = glm::vec<3, scalar_t>;
        using cov3_t = stroke::Cov3<scalar_t>;
        using Gaussian1d = stroke::gaussian::ParamsWithWeight<1, scalar_t>;

        whack::random::HostGenerator<scalar_t> rnd;

        for (int i = 0; i < 10; ++i) {
            const auto fun = [](const whack::Tensor<scalar_t, 1>& input) {
                const auto [g_pos, g_cov, r_pos, r_dir] = stroke::extract<vec3_t, cov3_t, vec3_t, vec3_t>(input);
                const auto g1d = stroke::gaussian::intersect_with_ray_inv_C<scalar_t>(g_pos, g_cov, { r_pos, r_dir });
                return stroke::pack_tensor<scalar_t>(g1d);
            };

            const auto fun_grad = [](const whack::Tensor<scalar_t, 1>& input, const whack::Tensor<scalar_t, 1>& grad_output) {
                const auto [g_pos, g_cov, r_pos, r_dir] = stroke::extract<vec3_t, cov3_t, vec3_t, vec3_t>(input);
                const auto grad_incoming = stroke::extract<Gaussian1d>(grad_output);

                const auto grad_outgoing = stroke::grad::gaussian::intersect_with_ray_inv_C<scalar_t>(g_pos, g_cov, { r_pos, r_dir }, grad_incoming);

                return stroke::pack_tensor<scalar_t>(grad_outgoing);
            };

            const auto g_pos = rnd.normal3() * 10.;
            const auto g_cov = inverse(stroke::host_random_cov<3, double>(&rnd));
            const auto r_pos = g_pos + rnd.normal3() * 0.5;
            const auto r_dir = normalize(rnd.normal3());
            const auto test_data = stroke::pack_tensor<scalar_t>(g_pos, g_cov, r_pos, r_dir);
            stroke::check_gradient(fun, fun_grad, test_data, scalar_t(0.000001));
        }
    }
}
