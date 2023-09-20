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
#include <glm/ext/scalar_constants.hpp>
#include <whack/random/generators.h>

#include "stroke/gaussian.h"
#include "stroke/ray.h"

#include "test_helpers.h"

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
        CHECK(gaussian::norm_factor(2.3f) == Catch::Approx(gaussian::norm_factor_inv_C(1 / 2.3)));

        // 2d
        CHECK(gaussian::norm_factor(Cov2(1.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi)));
        CHECK(gaussian::norm_factor(Cov2(4.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * 16)));
        CHECK(gaussian::norm_factor(Cov2(3.4, 1.2, 2.3)) == Catch::Approx(gaussian::norm_factor_inv_C(inverse(Cov2(3.4, 1.2, 2.3)))));

        // 3d
        CHECK(gaussian::norm_factor(Cov3(1.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * two_pi)));
        CHECK(gaussian::norm_factor(Cov3(4.f)) == Catch::Approx(1 / sqrt(two_pi * two_pi * two_pi * 4 * 4 * 4)));
        CHECK(gaussian::norm_factor(Cov3(3.4, 1.2, 1.3, 4.2, 1.0, 3.6)) == Catch::Approx(gaussian::norm_factor_inv_C(inverse(Cov3(3.4, 1.2, 1.3, 4.2, 1.0, 3.6)))));
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

    SECTION("integration")
    {
        // clang-format off
        CHECK(gaussian::integrate<float>(   0.f,    1.f, { -10000.f,   10000.f }) == Catch::Approx(1.0f));
        CHECK(gaussian::integrate<float>(   0.f,    1.f, {      0.f,   10000.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate<float>(   0.f,    1.f, { -10000.f,       0.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate<float>( 100.f,    1.f, { -10000.f,     100.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate<float>(   0.f,    1.f, {   1000.f,   10000.f }) == Catch::Approx(0.0f));
        CHECK(gaussian::integrate<float>(  12.f,    1.f, {     12.f,   10012.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate<float>(   0.f, 1000.f, {      1.f, 1000000.f }) == Catch::Approx(0.5f).margin(0.02));

        // https://www.wolframalpha.com/input?i=integrate+PDF%5BNormalDistribution%5B7%2C+5%5D%2C+x%5D+from+2+to+5&lang=es
        CHECK(gaussian::integrate<float>(   7.f, 5.f, {         2.f,       7.f }) == Catch::Approx(0.341345f));
        CHECK(gaussian::integrate<float>(   7.f, 5.f, {         2.f,       5.f }) == Catch::Approx(0.185923f));
        CHECK(gaussian::integrate<float>(  -3.f, 2.f, {        -1.f,       4.f }) == Catch::Approx(0.158423f));
        // clang-format on
    }
}
