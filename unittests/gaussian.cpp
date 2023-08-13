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

#include <whack/random/generators.h>

#include "glm/ext/scalar_constants.hpp"
#include "stroke/gaussian.h"
#include "stroke/ray.h"

namespace {
template <glm::length_t n_dims, typename scalar_t>
glm::mat<n_dims, n_dims, scalar_t> random_matrix(whack::random::HostGenerator<scalar_t>* rnd)
{
    glm::mat<n_dims, n_dims, scalar_t> mat;
    for (auto c = 0; c < n_dims; ++c) {
        for (auto r = 0; r < n_dims; ++r) {
            mat[c][r] = rnd->normal();
        }
    }
    return mat;
}

template <glm::length_t n_dims, typename scalar_t>
stroke::Cov<n_dims, scalar_t> random_cov(whack::random::HostGenerator<scalar_t>* rnd)
{
    const auto mat = random_matrix<n_dims, scalar_t>(rnd);
    return stroke::Cov<n_dims, scalar_t>(mat * transpose(mat)) + stroke::Cov<n_dims, scalar_t>(0.05);
}

} // namespace

TEST_CASE("gaussian")
{
    using namespace stroke;
    SECTION("normal cov matrix")
    {
        CHECK(gaussian::eval_exponential(0.f, 1.f, 0.f) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential(6.f, 1.f, 6.f) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential(0.f, 1.f, -1.5f) < 0.5);
        CHECK(gaussian::eval_exponential(10.f, 1.f, 1.f) < 0.1);

        CHECK(gaussian::eval_exponential<2, float>({ 0.f, 0.f }, { 1.f, 0.f, 1.f }, { 0.f, 0.f }) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential<2, float>({ 20.f, 30.f }, { 10.f, -8.f, 10.f }, { 20.f, 30.f }) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential<2, float>({ 1.f, 1.f }, { 1.f, 0.f, 1.f }, { 0.f, 0.f }) < 0.5f);
        CHECK(gaussian::eval_exponential<2, float>({ 0.f, 0.f }, { 1.f, 0.f, 1.f }, { 10.f, -10.f }) < 0.1f);

        CHECK(gaussian::eval_exponential<3, float>({ 0.f, 0.f, 0.f }, Cov3(1.0f), { 0.f, 0.f, 0.f }) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential<3, float>({ 10.f, 20.f, 30.f }, Cov3(2.0f), { 10.f, 20.f, 30.f }) == Catch::Approx(1.0));
        CHECK(gaussian::eval_exponential<3, float>({ -2.f, 1.f, 2.5f }, Cov3(1.5f), { 0.f, 0.f, 0.f }) < 0.5f);
        CHECK(gaussian::eval_exponential<3, float>({ 0.f, 0.f, 0.f }, Cov3(3.1f), { 10.f, -10.f, 0.f }) < 0.1f);
    }
    SECTION("inversed cov matrix")
    {
        whack::random::HostGenerator<float> rnd;
        for (int i = 0; i < 10; ++i) {
            {
                // 1d
                const auto centre = rnd.normal();
                const auto variance = rnd.normal();
                const auto point = rnd.normal();
                CHECK(gaussian::eval_exponential(centre, variance, point) == Catch::Approx(gaussian::eval_exponential_inv_C(centre, 1 / variance, point)));
            }
            {
                // 2d
                const auto centre = rnd.normal2();
                const auto covariance = random_cov<2, float>(&rnd);
                const auto point = rnd.normal2();
                CHECK(gaussian::eval_exponential(centre, covariance, point) == Catch::Approx(gaussian::eval_exponential_inv_C(centre, inverse(covariance), point)));
            }
            {
                // 3d
                const auto centre = rnd.normal3();
                const auto covariance = random_cov<3, float>(&rnd);
                const auto point = rnd.normal3();
                CHECK(gaussian::eval_exponential(centre, covariance, point) == Catch::Approx(gaussian::eval_exponential_inv_C(centre, inverse(covariance), point)));
            }
        }
    }

    SECTION("normalisation factor")
    {
        const auto two_pi = 2 * glm::pi<float>();
        // 1d
        CHECK(gaussian::norm_factor(1.f) == Catch::Approx(1 / sqrt(2 * glm::pi<float>())));
        CHECK(gaussian::norm_factor(4.f) == Catch::Approx(0.5f / sqrt(2 * glm::pi<float>())));

        // 2d
        CHECK(gaussian::norm_factor(Cov2(1.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi)));
        CHECK(gaussian::norm_factor(Cov2(4.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * 16)));

        // 3d
        CHECK(gaussian::norm_factor(Cov3(1.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * two_pi)));
        CHECK(gaussian::norm_factor(Cov3(4.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * two_pi * 4 * 4 * 4)));
    }

    SECTION("project a 3d gaussian onto a ray")
    {
        whack::random::HostGenerator<float> rnd;
        for (int i = 0; i < 10; ++i) {
            {
                const auto centre = rnd.normal3();
                const auto covariance = random_cov<3, float>(&rnd);

                {
                    const auto ray = stroke::Ray<3, float> { centre, glm::normalize(rnd.normal3()) };
                    const auto [oneD_weight, oneD_centre, oneD_variance] = gaussian::project_on_ray(centre, covariance, ray);
                    CHECK(oneD_centre == Catch::Approx(0));
                    CHECK(oneD_weight == Catch::Approx(1));
                    CHECK(gaussian::eval_exponential(centre, covariance, ray.origin + ray.direction * 0.8f) == Catch::Approx(oneD_weight * gaussian::eval_exponential(oneD_centre, oneD_variance, 0.8f)));
                }
                {
                    const auto direction = glm::normalize(rnd.normal3());
                    const auto ray = stroke::Ray<3, float> { centre - direction * 0.7f, direction };
                    const auto [oneD_weight, oneD_centre, oneD_variance] = gaussian::project_on_ray(centre, covariance, ray);
                    CHECK(oneD_centre == Catch::Approx(0.7));
                    CHECK(gaussian::eval_exponential(centre, covariance, ray.origin + ray.direction * 0.5f) == Catch::Approx(oneD_weight * gaussian::eval_exponential(oneD_centre, oneD_variance, 0.5f)));
                    CHECK(gaussian::eval_exponential(centre, covariance, ray.origin + ray.direction * 0.8f) == Catch::Approx(oneD_weight * gaussian::eval_exponential(oneD_centre, oneD_variance, 0.8f)));
                    CHECK(gaussian::eval_exponential(centre, covariance, ray.origin + ray.direction * 1.3f) == Catch::Approx(oneD_weight * gaussian::eval_exponential(oneD_centre, oneD_variance, 1.3f)));
                }
                {
                    const auto direction = glm::normalize(rnd.normal3());
                    const auto ray = stroke::Ray<3, float> { centre - direction * 0.7f, direction };
                    const auto cov_based = gaussian::project_on_ray(centre, covariance, ray);
                    const auto invcov_based = gaussian::project_on_ray_inv_C(centre, inverse(covariance), ray);
                    CHECK(cov_based.weight == Catch::Approx(invcov_based.weight));
                    CHECK(cov_based.centre == Catch::Approx(invcov_based.centre));
                    CHECK(cov_based.C == Catch::Approx(invcov_based.C));
                }
            }
        }
    }
}

// next steps:
// port existing gaussian stuff
// port existing welford stuff
// port gradients
// make everything cudaable
