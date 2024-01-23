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

#include "stroke/cuda_compat.h"
#include <cassert>

namespace stroke::grad {

template <typename A1, typename A2>
struct [[nodiscard]] TwoGrads {
    A1 m_left;
    A2 m_right;

    STROKE_DEVICES_INLINE
    void addTo(A1* left, A2* right)
    {
        *left += m_left;
        *right += m_right;
    }

    STROKE_DEVICES_INLINE
    void addTo(A1* left, bool right)
    {
        assert(right == false);
        *left += m_left;
    }

    STROKE_DEVICES_INLINE
    void addTo(bool left, A2* right)
    {
        assert(left == false);
        *right += m_right;
    }

    STROKE_DEVICES_INLINE
    A1 left() const
    {
        return m_left;
    }

    STROKE_DEVICES_INLINE
    A2 right() const
    {
        return m_right;
    }
};

template <typename A1, typename A2, typename A3>
struct [[nodiscard]] ThreeGrads {
    A1 m_left;
    A2 m_middle;
    A3 m_right;

    STROKE_DEVICES_INLINE
    void addTo(A1* left, A2* middle, A3* right)
    {
        if (left != nullptr)
            *left += m_left;
        if (middle != nullptr)
            *middle += m_middle;
        if (right != nullptr)
            *right += m_right;
    }

    STROKE_DEVICES_INLINE
    A1 left() const
    {
        return m_left;
    }

    STROKE_DEVICES_INLINE
    A2 middle() const
    {
        return m_middle;
    }

    STROKE_DEVICES_INLINE
    A3 right() const
    {
        return m_right;
    }
};

template <typename A1, typename A2, typename A3, typename A4>
struct [[nodiscard]] FourGrads {
    A1 m_first;
    A2 m_second;
    A3 m_third;
    A4 m_fourth;

    STROKE_DEVICES_INLINE void addTo(A1* first, A2* second, A3* third, A4* fourth)
    {
        if (first != nullptr)
            *first += m_first;
        if (second != nullptr)
            *second += m_second;
        if (third != nullptr)
            *third += m_third;
        if (fourth != nullptr)
            *fourth += m_fourth;
    }

    STROKE_DEVICES_INLINE A1 first() const
    {
        return m_first;
    }

    STROKE_DEVICES_INLINE A2 second() const
    {
        return m_second;
    }

    STROKE_DEVICES_INLINE A3 third() const
    {
        return m_third;
    }

    STROKE_DEVICES_INLINE A4 fourth() const
    {
        return m_fourth;
    }
};

} // namespace stroke::grad
