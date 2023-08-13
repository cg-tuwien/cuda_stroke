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

#include <cuda/std/array>
#include <cuda/std/functional>
#include <whack/array.h>

#include "stroke/algorithms.h"

TEST_CASE("algorithms")
{
    SECTION("transform cuda::std::array")
    {
        cuda::std::array<int, 3> m1 = { 1, 2, 3 };
        cuda::std::array<int, 3> result = stroke::transform(m1, [](int v) { return v * 10; });
        REQUIRE(result[0] == 10);
        REQUIRE(result[1] == 20);
        REQUIRE(result[2] == 30);
    }

    SECTION("transform whack::Array")
    {
        whack::Array<int, 3> m1 = { 1, 2, 3 };
        whack::Array<int, 3> result = stroke::transform(m1, [](int v) { return v * 10; });
        REQUIRE(result[0] == 10);
        REQUIRE(result[1] == 20);
        REQUIRE(result[2] == 30);
    }

    SECTION("transform_inplace cuda::std::array")
    {
        cuda::std::array<int, 3> arr = { 1, 2, 3 };
        stroke::transform_inplace(&arr, [](int v) { return v * 10; });
        REQUIRE(arr[0] == 10);
        REQUIRE(arr[1] == 20);
        REQUIRE(arr[2] == 30);
    }

    SECTION("cwise fun array on array cuda::std::array")
    {
        cuda::std::array<int, 3> m1 = { 1, 2, 3 };
        cuda::std::array<int, 3> m2 = { 10, 20, 30 };
        cuda::std::array<int, 3> result = stroke::cwise_fun(m1, m2, [](auto a, auto b) { return a + b; });
        REQUIRE(result[0] == 11);
        REQUIRE(result[1] == 22);
        REQUIRE(result[2] == 33);
    }

    SECTION("cwise fun array on array whack::Array")
    {
        whack::Array<int, 3> m1 = { 1, 2, 3 };
        whack::Array<int, 3> m2 = { 10, 20, 30 };
        whack::Array<int, 3> result = stroke::cwise_fun(m1, m2, cuda::std::plus<int>());
        REQUIRE(result[0] == 11);
        REQUIRE(result[1] == 22);
        REQUIRE(result[2] == 33);
    }

    SECTION("cwise_inplace fun array on array whack::Array")
    {
        whack::Array<int, 3> m1 = { 1, 2, 3 };
        whack::Array<int, 3> m2 = { 10, 20, 30 };
        stroke::cwise_inplace_fun(&m1, m2, cuda::std::plus<int>());
        REQUIRE(m1[0] == 11);
        REQUIRE(m1[1] == 22);
        REQUIRE(m1[2] == 33);
    }

    SECTION("cwise fun array on scalar cuda::std::array")
    {
        cuda::std::array<float, 3> m1 = { 1.9, 2.2, 3.5 };
        cuda::std::array<int, 3> result = stroke::cwise_fun(m1, 10.f, cuda::std::multiplies<int>()); // carefull with implicit casting!
        REQUIRE(result[0] == 10);
        REQUIRE(result[1] == 20);
        REQUIRE(result[2] == 30);
    }

    SECTION("cwise fun array on scalar whack::Array")
    {
        whack::Array<float, 3> m1 = { 1.9, 2.2, 3.5 };
        whack::Array<float, 3> result = stroke::cwise_fun(m1, 10.f, cuda::std::multiplies<>());
        REQUIRE(result[0] == Catch::Approx(19));
        REQUIRE(result[1] == Catch::Approx(22));
        REQUIRE(result[2] == Catch::Approx(35));
    }

    SECTION("cwise_inplace fun array on scalar whack::Array")
    {
        whack::Array<float, 3> m1 = { 1.9, 2.2, 3.5 };
        stroke::cwise_inplace_fun(&m1, 10.f, cuda::std::multiplies<>());
        REQUIRE(m1[0] == Catch::Approx(19));
        REQUIRE(m1[1] == Catch::Approx(22));
        REQUIRE(m1[2] == Catch::Approx(35));
    }

    SECTION("reduce fun on whack::Array")
    {
        {
            whack::Array<int, 3> m1 = { 1, 2, 3 };
            int result = stroke::reduce(m1, int(0), cuda::std::plus<int>());
            REQUIRE(result == 6);
        }
        {
            whack::Array<int, 3> m1 = { 1, 2, 3 };
            bool result = stroke::reduce(m1, int(1), cuda::std::logical_and<int>());
            REQUIRE(result);
        }
        {
            whack::Array<int, 3> m1 = { 0, 2, 3 };
            bool result = stroke::reduce(m1, int(1), cuda::std::logical_and<int>());
            REQUIRE(!result);
        }
    }
}
