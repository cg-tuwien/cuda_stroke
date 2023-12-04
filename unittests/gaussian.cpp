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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <whack/random/generators.h>

#include "stroke/gaussian.h"
#include "stroke/ray.h"

#include "test_helpers.h"

TEST_CASE("stroke gaussian")
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

    SECTION("normalised gaussian")
    {
        // 1d https://www.wolframalpha.com/input?i2d=true&i=pdf%5C%2840%29N%5C%2840%290.4%5C%2844%29+1.3%5C%2841%29%5C%2844%29+0.8%5C%2841%29
        CHECK(gaussian::eval_normalised(0.4, 1.3, 0.8) == Catch::Approx(0.329013));

        // 2d and 3d https://www.wolframcloud.com/obj/284b53f9-0a2f-4414-94af-d30e68b7bed2
        // same, but screenshot: https://imgur.com/a/N4AJG0P
        CHECK(gaussian::eval_normalised<2, double>({ 0.4, 0.7 }, { 1.5, 0.3, 1.2 }, { 0.3, 0.9 }) == Catch::Approx(0.118756));
        CHECK(gaussian::eval_normalised<3, double>({ 0.4, 0.7, 0.8 }, { 1.5, 0.3, 0.2, 0.9, 0.1, 1.3 }, { 0.3, 0.9, 0.7 }) == Catch::Approx(0.0484231));
    }

    SECTION("intersect a 3d gaussian with a ray")
    {
        whack::random::HostGenerator<float> rnd;
        for (int i = 0; i < 10; ++i) {
            {
                const auto centre = rnd.normal3();
                const auto covariance = random_cov<3, float>(&rnd);

                {
                    const auto ray = stroke::Ray<3, float> { centre, glm::normalize(rnd.normal3()) };
                    const auto [oneD_weight, oneD_centre, oneD_variance] = gaussian::intersect_with_ray(centre, covariance, ray);
                    CHECK(oneD_centre == Catch::Approx(0));
                    CHECK(oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 0.f) == Catch::Approx(gaussian::eval_normalised(centre, covariance, centre)));
                    CHECK(gaussian::eval_normalised(centre, covariance, ray.origin + ray.direction * 0.8f) == Catch::Approx(oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 0.8f)));
                }
                {
                    const auto direction = glm::normalize(rnd.normal3());
                    const auto ray = stroke::Ray<3, float> { centre - direction * 0.7f, direction };
                    const auto [oneD_weight, oneD_centre, oneD_variance] = gaussian::intersect_with_ray(centre, covariance, ray);
                    CHECK(oneD_centre == Catch::Approx(0.7));
                    CHECK(gaussian::eval_normalised(centre, covariance, ray.origin + ray.direction * 0.5f) == Catch::Approx(oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 0.5f)));
                    CHECK(gaussian::eval_normalised(centre, covariance, ray.origin + ray.direction * 0.8f) == Catch::Approx(oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 0.8f)));
                    CHECK(gaussian::eval_normalised(centre, covariance, ray.origin + ray.direction * 1.3f) == Catch::Approx(oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 1.3f)));
                    CHECK(0.4f * gaussian::eval_normalised(centre, covariance, ray.origin + ray.direction * 1.3f) == Catch::Approx(0.4f * oneD_weight * gaussian::eval_normalised(oneD_centre, oneD_variance, 1.3f)));
                }
                {
                    const auto direction = glm::normalize(rnd.normal3());
                    const auto ray = stroke::Ray<3, float> { centre - direction * 0.7f, direction };
                    const auto cov_based = gaussian::intersect_with_ray(centre, covariance, ray);
                    const auto invcov_based = gaussian::intersect_with_ray_inv_C(centre, inverse(covariance), ray);
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
        CHECK(gaussian::integrate_var<float>(   0.f,    1.f, { -10000.f,   10000.f }) == Catch::Approx(1.0f));
        CHECK(gaussian::integrate_var<float>(   0.f,    1.f, {      0.f,   10000.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate_var<float>(   0.f,    1.f, { -10000.f,       0.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate_var<float>( 100.f,    1.f, { -10000.f,     100.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate_var<float>(   0.f,    1.f, {   1000.f,   10000.f }) == Catch::Approx(0.0f));
        CHECK(gaussian::integrate_var<float>(  12.f,    1.f, {     12.f,   10012.f }) == Catch::Approx(0.5f));
        CHECK(gaussian::integrate_var<float>(   0.f, 1000.f, {      1.f, 1000000.f }) == Catch::Approx(0.5f).margin(0.02));

        // https://www.wolframalpha.com/input?i=integrate+PDF%5BNormalDistribution%5B7%2C+5%5D%2C+x%5D+from+2+to+5&lang=es
        CHECK(gaussian::integrate_SD<float>(   7.f, 5.f, {         2.f,       7.f }) == Catch::Approx(0.341345f));
        CHECK(gaussian::integrate_SD<float>(   7.f, 5.f, {         2.f,       5.f }) == Catch::Approx(0.185923f));
        CHECK(gaussian::integrate_SD<float>(  -3.f, 2.f, {        -1.f,       4.f }) == Catch::Approx(0.158423f));

        CHECK(gaussian::integrate_inv_SD<float>(   7.f, 1/5.f, {         2.f,       7.f }) == Catch::Approx(0.341345f));
        CHECK(gaussian::integrate_inv_SD<float>(   7.f, 1/5.f, {         2.f,       5.f }) == Catch::Approx(0.185923f));
        CHECK(gaussian::integrate_inv_SD<float>(  -3.f, 1/2.f, {        -1.f,       4.f }) == Catch::Approx(0.158423f));
        // clang-format on
    }
}
