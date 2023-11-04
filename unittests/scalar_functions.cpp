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

#include <stroke/scalar_functions.h>

TEST_CASE("stroke scalar functions")
{

    SECTION("transform cuda::std::array")
    {
        CHECK(stroke::clamp(0, 1, 10) == 1);
        CHECK(stroke::clamp(5, 1, 10) == 5);
        CHECK(stroke::clamp(1, 1, 10) == 1);
        CHECK(stroke::clamp(10, 1, 10) == 10);
        CHECK(stroke::clamp(11, 1, 10) == 10);

        CHECK(stroke::clamp(0.0, 0.1, 0.9) == 0.1);
        CHECK(stroke::clamp(0.1f, 0.1f, 0.9f) == 0.1f);
        CHECK(stroke::clamp(3.0f, 0.1f, 0.9f) == 0.9f);
    }
}
