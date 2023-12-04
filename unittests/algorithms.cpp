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

#include <cuda/std/array>
#include <cuda/std/functional>
#include <whack/array.h>

#include "stroke/algorithms.h"

TEST_CASE("stroke algorithms")
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
