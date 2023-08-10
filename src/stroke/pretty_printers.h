/*****************************************************************************
 * Alpine Terrain Renderer
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

#pragma once

#include <ostream>

#include <glm/gtx/string_cast.hpp>

#include "matrix.h"

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
