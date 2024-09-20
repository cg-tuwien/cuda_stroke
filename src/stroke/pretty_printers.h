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

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/string_cast.hpp>
#include <ostream>
#include <sstream>
#include <stroke/detail/symmetric_mat.h>

template <glm::length_t n_dims, typename T>
std::ostream& operator<<(std::ostream& os, const glm::vec<n_dims, T>& v)
{
    os << glm::to_string(v);
    return os;
}

template <glm::length_t n, glm::length_t m, typename T>
std::ostream& operator<<(std::ostream& os, const glm::mat<n, m, T>& mat)
{
    os << glm::to_string(mat);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const stroke::Cov2<T>& m)
{
    os << "Cov2((" << m[0] << ", " << m[1] << "), (" << m[1] << ", " << m[2] << "))";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const stroke::Cov3<T>& m)
{
    os << "Cov3((" << m[0] << ", " << m[1] << ", " << m[2] << "), (" << m[1] << ", " << m[3] << ", " << m[4] << "), (" << m[2] << ", " << m[4] << ", " << m[5] << "))";
    return os;
}

// must go below the stream operators (and i don't know why, but printing glm values stops working if this is above)
#include <catch2/matchers/catch_matchers.hpp>

template <glm::length_t N, typename scalar>
class VecMatcher : public Catch::Matchers::MatcherBase<glm::vec<N, scalar>> {
    using Vec = glm::vec<N, scalar>;
    Vec expected;
    scalar epsilon;

public:
    VecMatcher(const Vec& expected, scalar epsilon = 1e-5)
        : expected(expected)
        , epsilon(epsilon)
    {
    }

    // Perform the fuzzy comparison
    bool match(const Vec& actual) const override
    {
        return glm::all(glm::epsilonEqual(expected, actual, epsilon));
    }

    // Produce a detailed message when the test fails
    std::string describe() const override
    {
        std::ostringstream ss;
        ss << "is approximately equal to (" << expected << ") with epsilon = " << epsilon;
        return ss.str();
    }
};
