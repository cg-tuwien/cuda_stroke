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

#include "stroke/pretty_printers.h" // must come before catch includes

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_helpers.h"

TEST_CASE("stroke matrices")
{
    SECTION("SymmetricMat/Cov construction")
    {
        // 2d
        CHECK(stroke::Cov<2, float>(1.f, 2.f, 3.f) == stroke::Cov2(1.f, 2.f, 3.f));
        CHECK(stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f) == stroke::Cov2(1.f, 2.f, 3.f));
        CHECK(stroke::Cov2<float>() == stroke::Cov2(0.f, 0.f, 0.f));
        CHECK(stroke::Cov2(glm::mat2()) == stroke::Cov2(0.f, 0.f, 0.f));
        CHECK(stroke::Cov2(glm::mat2(1, 2, 2, 3)) == stroke::Cov2(1.f, 2.f, 3.f));
        CHECK(stroke::Cov2(2.f) == stroke::Cov2(2.f, 0.f, 2.f));
        CHECK(stroke::Cov2<float>(stroke::Cov2(2.f).data()) == stroke::Cov2(2.f));
        CHECK(stroke::Cov2(stroke::Cov2(2.f)) == stroke::Cov2(2.f)); // copy ctor

        // 3d
        CHECK(stroke::Cov<3, float>(1, 2, 3, 4, 5, 6) == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
        CHECK(stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6) == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
        CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f));
        CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
        CHECK(stroke::Cov3(2.f) == stroke::Cov3(2.f, 0.f, 0.f, 2.f, 0.f, 2.f));
        CHECK(stroke::Cov3<float>(stroke::Cov3(2.f).data()) == stroke::Cov3(2.f));
        CHECK(stroke::Cov3(stroke::Cov3(2.f)) == stroke::Cov3(2.f)); // copy ctor
        CHECK(stroke::Cov3(glm::mat3(
                  1, 2, 3,
                  2, 4, 5,
                  3, 5, 6))
            == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
    }

    SECTION("assignment and copy construction")
    {
        stroke::Cov<2, float> a;
        a = stroke::Cov<2, float>(1.f, 2.f, 3.f);
        a = stroke::Cov2<float>(1.f, 2.f, 3.f);
        a = stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f);

        stroke::Cov<3, float> b;
        b = stroke::Cov<3, float>(1, 2, 3, 4, 5, 6);
        b = stroke::Cov3<float>(1, 2, 3, 4, 5, 6);
        b = stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6);

        stroke::Cov2<float> c;
        c = stroke::Cov<2, float>(1.f, 2.f, 3.f);
        c = stroke::Cov2<float>(1.f, 2.f, 3.f);
        c = stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f);

        stroke::Cov3<float> d;
        d = stroke::Cov<3, float>(1, 2, 3, 4, 5, 6);
        d = stroke::Cov3<float>(1, 2, 3, 4, 5, 6);
        d = stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6);

        stroke::Cov<2, float>(stroke::Cov<2, float>(1.f, 2.f, 3.f));
        stroke::Cov<2, float>(stroke::Cov2<float>(1.f, 2.f, 3.f));
        stroke::Cov<2, float>(stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f));

        stroke::Cov<3, float>(stroke::Cov<3, float>(1, 2, 3, 4, 5, 6));
        stroke::Cov<3, float>(stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
        stroke::Cov<3, float>(stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6));

        stroke::Cov2<float>(stroke::Cov<2, float>(1.f, 2.f, 3.f));
        stroke::Cov2<float>(stroke::Cov2<float>(1.f, 2.f, 3.f));
        stroke::Cov2<float>(stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f));

        stroke::Cov3<float>(stroke::Cov<3, float>(1, 2, 3, 4, 5, 6));
        stroke::Cov3<float>(stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
        stroke::Cov3<float>(stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6));
    }

    SECTION("data type casting")
    {
        stroke::Cov<2, float>(stroke::Cov<2, double>(1));
        stroke::Cov<2, double>(stroke::Cov<2, float>(1));
        stroke::Cov<3, float>(stroke::Cov<3, double>(1));
        stroke::Cov<3, double>(stroke::Cov<3, float>(1));

        stroke::Cov2<float>(stroke::Cov2<double>(1));
        stroke::Cov2<double>(stroke::Cov2<float>(1));
        stroke::Cov3<float>(stroke::Cov3<double>(1));
        stroke::Cov3<double>(stroke::Cov3<float>(1));
    }

    SECTION("SymmetricMat/Cov glm casting")
    {
        CHECK(glm::mat2(1, 2, 2, 3) == glm::mat2(stroke::Cov2(1.f, 2.f, 3.f)));
        CHECK(glm::mat3(
                  1, 2, 3,
                  2, 4, 5,
                  3, 5, 6)
            == glm::mat3(stroke::Cov3<float>(1, 2, 3, 4, 5, 6)));

        CHECK(glm::mat2(1, 2, 2, 3) == to_glm(stroke::Cov2(1.f, 2.f, 3.f)));
        CHECK(glm::mat3(
                  1, 2, 3,
                  2, 4, 5,
                  3, 5, 6)
            == to_glm(stroke::Cov3<float>(1, 2, 3, 4, 5, 6)));
    }

    SECTION("SymmetricMat/Cov element access")
    {
        {
            const auto cov = stroke::Cov2(1, 2, 3);
            CHECK(cov[0] == 1);
            CHECK(cov[1] == 2);
            CHECK(cov[2] == 3);

            CHECK(cov(0, 0) == 1);
            CHECK(cov(0, 1) == 2);
            CHECK(cov(1, 0) == 2);
            CHECK(cov(1, 1) == 3);
        }

        {
            const auto cov = stroke::Cov3<float>(1, 2, 3, 4, 5, 6);
            CHECK(cov[0] == 1.f);
            CHECK(cov[1] == 2.f);
            CHECK(cov[2] == 3.f);
            CHECK(cov[3] == 4.f);
            CHECK(cov[4] == 5.f);
            CHECK(cov[5] == 6.f);
            CHECK(cov(0, 0) == 1);
            CHECK(cov(0, 2) == 3);
            CHECK(cov(2, 2) == 6);
            const auto glmcov = glm::mat3(cov);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    CHECK(cov(row, col) == glmcov[row][col]);
                }
            }
        }
    }

    SECTION("SymmetricMat/Cov mul with vector")
    {
        const auto check = [](const auto& cov, const auto& vec) {
            const auto glmcov = to_glm(cov);
            CHECK(cov * vec == glmcov * vec);
        };
        check(stroke::Cov2(1.f, 2.f, 3.f), glm::vec2(1.f, 2.f));
        check(stroke::Cov2(2.f, 1.5f, 4.f), glm::vec2(3.f, 2.f));

        check(stroke::Cov3(1.f, 2.f, 3.f, 4.f, 5.f, 6.f), glm::vec3(1.f, 2.f, 3.f));
        check(stroke::Cov3(5.3f, 2.2f, 1.5f, 3.4f, 2.3f, 6.5f), glm::vec3(3.3f, 2.3f, 1.2f));
    }

    SECTION("SymmetricMat/Cov basic matrix ops")
    {
        const auto check = [](const auto& A, const auto& B) {
            const auto glmA = to_glm(A);
            const auto glmB = to_glm(B);
            CHECK(to_glm(A + B) == glmA + glmB);
            CHECK(to_glm(A - B) == glmA - glmB);

            CHECK(to_glm(A + 2.f) == glmA + 2.f);
            CHECK(to_glm(A - 2.f) == glmA - 2.f);
            CHECK(to_glm(A * 2.f) == glmA * 2.f);
            CHECK(to_glm(2.f + A) == 2.f + glmA);
            CHECK(to_glm(2.f - A) == 2.f - glmA);
            CHECK(to_glm(2.f * A) == 2.f * glmA);
            CHECK(to_glm(matrixCompMult(A, B)) == matrixCompMult(glmA, glmB));
            CHECK(A * B == glmA * glmB);
            CHECK(determinant(A) == Catch::Approx(determinant(glmA)));
            CHECK(determinant(B) == Catch::Approx(determinant(glmB)));
            CHECK(isnan(A) == false);
            auto C = A;
            CHECK(!stroke::isnan(A));
            CHECK(!stroke::isnan(C));
            C[0] = std::numeric_limits<double>::quiet_NaN();
            CHECK(stroke::isnan(C[0]));
            CHECK(stroke::isnan(C));
            C[0] = 1.0;
            CHECK(!stroke::isnan(C));
            C[1] = 0 / 0.0;
            CHECK(stroke::isnan(C));
        };
        check(stroke::Cov2(4.f, 2.f, 3.f), stroke::Cov2(1.3f, 0.8f, 2.3f));
        check(stroke::Cov2(2.f, 1.5f, 4.f), stroke::Cov2(1.8f, 0.8f, 2.3f));

        check(stroke::Cov3(11.f, 2.f, 3.f, 14.f, 5.f, 16.f), stroke::Cov3(5.4f, 0.8f, 3.5f, 6.2f, 2.5f, 9.6f));
        check(stroke::Cov3(5.3f, 2.2f, 1.5f, 3.4f, 2.3f, 6.5f), stroke::Cov3(5.4f, 1.8f, 0.5f, 4.2f, 1.3f, 7.6f));
    }

    SECTION("SymmetricMat/Cov inverse")
    {
        const auto check = [](const auto& S) {
            const auto glm_S = to_glm(S);
            REQUIRE(det(S) > 0.1);
            const auto inv_S = inverse(S);
            const auto glm_inv_S = inverse(glm_S);
            for (auto col = 0; col < glm_S.length(); ++col) {
                for (auto row = 0; row < glm_S.length(); ++row) {
                    CHECK(inv_S(col, row) == Catch::Approx(glm_inv_S[col][row]));
                }
            }
        };
        check(stroke::Cov2(4.f, 2.f, 3.f));
        check(stroke::Cov2(1.3f, 0.8f, 2.3f));
        check(stroke::Cov2(2.f, 1.5f, 4.f));
        check(stroke::Cov2(1.8f, 0.8f, 2.3f));

        check(stroke::Cov3(11.f, 2.f, 3.f, 14.f, 5.f, 16.f));
        check(stroke::Cov3(5.4f, 0.8f, 3.5f, 6.2f, 2.5f, 9.6f));
        check(stroke::Cov3(5.3f, 2.2f, 1.5f, 3.4f, 2.3f, 6.5f));
        check(stroke::Cov3(5.4f, 1.8f, 0.5f, 4.2f, 1.3f, 7.6f));
    }

    SECTION("SymmetricMat/Cov affine transform")
    {
        whack::random::HostGenerator<float> rng;

        const auto check = [](const auto& S, const auto& M) {
            const auto glm_S = to_glm(S);
            REQUIRE(det(S) > 0.1);

            const auto spoke_result = affine_transform(S, M);
            const auto glm_result = M * glm_S * transpose(M);

            for (auto col = 0; col < glm_S.length(); ++col) {
                for (auto row = 0; row < glm_S.length(); ++row) {
                    CHECK(spoke_result(col, row) == Catch::Approx(glm_result[col][row]).scale(1));
                }
            }
        };
        for (int i = 0; i < 10; ++i) {
            check(random_cov<2, float>(&rng), random_matrix<2, float>(&rng));
            check(random_cov<3, float>(&rng), random_matrix<3, float>(&rng));
        }
    }
}
