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

#include "stroke/pretty_printers.h"

#include <catch2/catch_test_macros.hpp>

#include "stroke/matrix.h"
#include "stroke/matrix_functions.h"

TEST_CASE("matrices: SymmetricMat/Cov construction")
{
    // 2d
    CHECK(stroke::Cov<2, float>(1.f, 2.f, 3.f) == stroke::Cov2(1.f, 2.f, 3.f));
    CHECK(stroke::SymmetricMat<2, float>(1.f, 2.f, 3.f) == stroke::Cov2(1.f, 2.f, 3.f));
    CHECK(stroke::Cov2<float>() == stroke::Cov2(0.f, 0.f, 0.f));
    CHECK(stroke::Cov2(glm::mat2()) == stroke::Cov2(0.f, 0.f, 0.f));
    CHECK(stroke::Cov2(glm::mat2(1, 2, 2, 3)) == stroke::Cov2(1.f, 2.f, 3.f));
    CHECK(stroke::Cov2(2.f) == stroke::Cov2(2.f, 0.f, 2.f));

    // 3d
    CHECK(stroke::Cov<3, float>(1, 2, 3, 4, 5, 6) == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
    CHECK(stroke::SymmetricMat<3, float>(1, 2, 3, 4, 5, 6) == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
    CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f));
    CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
    CHECK(stroke::Cov3(2.f) == stroke::Cov3(2.f, 0.f, 0.f, 2.f, 0.f, 2.f));
    CHECK(stroke::Cov3(glm::mat3(
              1, 2, 3,
              2, 4, 5,
              3, 5, 6))
        == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
}

TEST_CASE("matrices: SymmetricMat/Cov glm casting")
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

TEST_CASE("matrices: SymmetricMat/Cov element access")
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

TEST_CASE("matrices: SymmetricMat/Cov mul with vector")
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

TEST_CASE("matrices: det")
{
    CHECK(determinant(glm::mat2(1, 2, 2, 3)) == det(stroke::Cov2(1.f, 2.f, 3.f)));
    CHECK(det(stroke::Cov<2, float>(2, 3, 4)) == -1);
    CHECK(det(stroke::Cov<2, float>(1, 0, 1)) == 1);
    CHECK(det(stroke::Cov<2, float>(2, 3, 4)) == -1);
}
