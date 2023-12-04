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

#include <random>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

#include "stroke/welford.h"

namespace {
std::pair<glm::dvec3, glm::dmat3x3> compute_moments(const std::vector<std::pair<double, glm::dvec3>>& data)
{
    double w_sum = 0;
    glm::dvec3 mean = {};
    for (const auto& d : data) {
        const auto w = d.first;
        const auto p = d.second;
        mean += w * p;
        w_sum += w;
    }
    mean /= w_sum;
    glm::dmat3x3 cov = {};
    for (const auto& d : data) {
        const auto w = d.first;
        const auto p = d.second;

        cov[0][0] += w * (p.x - mean.x) * (p.x - mean.x);
        cov[0][1] += w * (p.x - mean.x) * (p.y - mean.y);
        cov[0][2] += w * (p.x - mean.x) * (p.z - mean.z);
        cov[1][0] += w * (p.y - mean.y) * (p.x - mean.x);
        cov[1][1] += w * (p.y - mean.y) * (p.y - mean.y);
        cov[1][2] += w * (p.y - mean.y) * (p.z - mean.z);
        cov[2][0] += w * (p.z - mean.z) * (p.x - mean.x);
        cov[2][1] += w * (p.z - mean.z) * (p.y - mean.y);
        cov[2][2] += w * (p.z - mean.z) * (p.z - mean.z);
    }
    cov /= w_sum;
    return { mean, cov };
}
double frobenius_norm(const glm::dmat3x3& m)
{
    double n = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            n += m[i][j] * m[i][j];
        }
    }
    return std::sqrt(n);
}
} // namespace

TEST_CASE("stroke welford")
{
    constexpr double mean_x = 42.0;
    constexpr double sd_x = 4.2;
    constexpr double sd_y = 0.5;
    constexpr unsigned N = 1000000;

    std::mt19937 gen;
    std::normal_distribution<double> d_x(mean_x, sd_x);
    std::normal_distribution<double> d_y(0.0, sd_y);
    std::uniform_real_distribution<double> d_w(0, 2);

    stroke::welford::WeightedMeanAndCov<3, double> welford_with_cov;
    stroke::welford::WeightedMean<double, glm::dvec3> welford_mean_only;

    SECTION("mean_and_cov")
    {
        std::vector<std::pair<double, glm::dvec3>> data;
        for (unsigned i = 0; i < N; ++i) {
            double x = d_x(gen);
            double y = -0.5 * x + d_y(gen);
            double z = x * 3.0;
            double w = std::max(0.05, d_w(gen) + x - 42);
            welford_with_cov.addValue(w, { x, y, z });
            welford_mean_only.addValue(w, { x, y, z });
            data.emplace_back(w, glm::dvec3(x, y, z));
        }
        const glm::dvec3 welford_mean = welford_with_cov.mean();
        const glm::dvec3 welford_mean2 = welford_mean_only.mean();
        const glm::dmat3x3 welford_cov = glm::dmat3x3(welford_with_cov.cov_matrix());
        glm::dvec3 actual_mean;
        glm::dmat3x3 actual_cov;
        std::tie(actual_mean, actual_cov) = compute_moments(data);

        CHECK(std::abs(glm::dot(welford_mean - actual_mean, welford_mean - actual_mean)) < 0.0000000000001);
        CHECK(std::abs(glm::dot(welford_mean2 - actual_mean, welford_mean2 - actual_mean)) < 0.0000000000001);
        const auto tmp = frobenius_norm(welford_cov - actual_cov);
        CHECK(std::abs(tmp) < 0.000000001);
    }
}
