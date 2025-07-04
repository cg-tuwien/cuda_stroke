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

#include <stroke/scalar_functions.h>

TEST_CASE("stroke scalar functions")
{

    SECTION("clamp")
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
