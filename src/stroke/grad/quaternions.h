/****************************************************************************
 *  Copyright (C) 2024 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
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
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

namespace stroke::grad {

template <typename Scalar>
STROKE_DEVICES_INLINE glm::qua<Scalar> toMat3(const glm::qua<Scalar>& quat, const glm::mat<3, 3, Scalar>& incoming_grad)
{
    Scalar qxx(quat.x * quat.x);
    Scalar qyy(quat.y * quat.y);
    Scalar qzz(quat.z * quat.z);
    Scalar qxz(quat.x * quat.z);
    Scalar qxy(quat.x * quat.y);
    Scalar qyz(quat.y * quat.z);
    Scalar qwx(quat.w * quat.x);
    Scalar qwy(quat.w * quat.y);
    Scalar qwz(quat.w * quat.z);

    Scalar grad_qxx = 0;
    Scalar grad_qyy = 0;
    Scalar grad_qzz = 0;
    Scalar grad_qxz = 0;
    Scalar grad_qxy = 0;
    Scalar grad_qyz = 0;
    Scalar grad_qwx = 0;
    Scalar grad_qwy = 0;
    Scalar grad_qwz = 0;

    // mat<3, 3, T, Q> Result(T(1));
    // Result[0][0] = T(1) - T(2) * (qyy +  qzz);
    grad_qyy += incoming_grad[0][0] * -2;
    grad_qzz += incoming_grad[0][0] * -2;

    // Result[0][1] = T(2) * (qxy + qwz);
    grad_qxy += incoming_grad[0][1] * 2;
    grad_qwz += incoming_grad[0][1] * 2;

    // Result[0][2] = T(2) * (qxz - qwy);
    grad_qxz += incoming_grad[0][2] * 2;
    grad_qwy += incoming_grad[0][2] * -2;

    // Result[1][0] = T(2) * (qxy - qwz);
    grad_qxy += incoming_grad[1][0] * 2;
    grad_qwz += incoming_grad[1][0] * -2;

    // Result[1][1] = T(1) - T(2) * (qxx +  qzz);
    grad_qxx += incoming_grad[1][1] * -2;
    grad_qzz += incoming_grad[1][1] * -2;

    // Result[1][2] = T(2) * (qyz + qwx);
    grad_qyz += incoming_grad[1][2] * 2;
    grad_qwx += incoming_grad[1][2] * 2;

    // Result[2][0] = T(2) * (qxz + qwy);
    grad_qxz += incoming_grad[2][0] * 2;
    grad_qwy += incoming_grad[2][0] * 2;

    // Result[2][1] = T(2) * (qyz - qwx);
    grad_qyz += incoming_grad[2][1] * 2;
    grad_qwx += incoming_grad[2][1] * -2;

    // Result[2][2] = T(1) - T(2) * (qxx +  qyy);
    grad_qxx += incoming_grad[2][2] * -2;
    grad_qyy += incoming_grad[2][2] * -2;
    // return Result;

    glm::qua<Scalar> grad_quat = { 0, 0, 0, 0 };
    // Scalar qxx(quat.x * quat.x);
    grad_quat.x += grad_qxx * quat.x * 2;

    // Scalar qyy(quat.y * quat.y);
    grad_quat.y += grad_qyy * quat.y * 2;

    // Scalar qzz(quat.z * quat.z);
    grad_quat.z += grad_qzz * quat.z * 2;

    // Scalar qxz(quat.x * quat.z);
    grad_quat.x += grad_qxz * quat.z;
    grad_quat.z += grad_qxz * quat.x;

    // Scalar qxy(quat.x * quat.y);
    grad_quat.x += grad_qxy * quat.y;
    grad_quat.y += grad_qxy * quat.x;

    // Scalar qyz(quat.y * quat.z);
    grad_quat.y += grad_qyz * quat.z;
    grad_quat.z += grad_qyz * quat.y;

    // Scalar qwx(quat.w * quat.x);
    grad_quat.w += grad_qwx * quat.x;
    grad_quat.x += grad_qwx * quat.w;

    // Scalar qwy(quat.w * quat.y);
    grad_quat.w += grad_qwy * quat.y;
    grad_quat.y += grad_qwy * quat.w;

    // Scalar qwz(quat.w * quat.z);
    grad_quat.w += grad_qwz * quat.z;
    grad_quat.z += grad_qwz * quat.w;

    return grad_quat;
}

template <typename Scalar>
STROKE_DEVICES_INLINE glm::qua<Scalar> toMat4(const glm::qua<Scalar>& quat, const glm::mat<4, 4, Scalar>& incoming_grad)
{
    return toMat3(quat, glm::mat<3, 3, Scalar>(incoming_grad));
}

} // namespace stroke::grad
