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

#include "matrix.h"

namespace stroke {

template <unsigned n_dims, typename scalar_t>
scalar_t det(const SymmetricMat<n_dims, scalar_t>& m)
{
    return {};
}

template <typename scalar_t>
scalar_t det(const SymmetricMat<2, scalar_t>& m)
{
    return m.data[0] * m.data[2] - sq(m.data[1]);
}

} // namespace stroke
