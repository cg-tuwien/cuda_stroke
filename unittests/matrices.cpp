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

TEST_CASE("matrix construction")
{
    // 2d construct
    CHECK(stroke::Cov2<float>() == stroke::Cov2(0.f, 0.f, 0.f));
    CHECK(stroke::Cov2(glm::mat2()) == stroke::Cov2(0.f, 0.f, 0.f));
    CHECK(stroke::Cov2(2.f) == stroke::Cov2(2.f, 0.f, 2.f));

    // 3d construct
    CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f));
    CHECK(stroke::Cov3<float>() == stroke::Cov3(0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
    CHECK(stroke::Cov3(2.f) == stroke::Cov3(2.f, 0.f, 0.f, 2.f, 0.f, 2.f));
    CHECK(stroke::Cov3(glm::mat3x3(1, 2, 3, 2, 4, 5, 3, 4, 6)) == stroke::Cov3<float>(1, 2, 3, 4, 5, 6));
}

TEST_CASE("det")
{
    CHECK(det(stroke::Cov<2, float>(2, 3, 4)) == -1);
    CHECK(det(stroke::Cov<2, float>(1, 0, 1)) == 1);
    CHECK(det(stroke::Cov<2, float>(2, 3, 4)) == -1);
}
