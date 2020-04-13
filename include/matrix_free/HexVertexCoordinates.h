// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef HEX_VERTEX_COORDINATES_H
#define HEX_VERTEX_COORDINATES_H

#include "matrix_free/LocalArray.h"
#include "Kokkos_Core.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
template <int p, typename InputArray>
KOKKOS_FUNCTION LocalArray<ftype[3][8]>
hex_vertex_coordinates(int index, const InputArray& xc)
{
  static_assert(InputArray::Rank == 5, "");
  LocalArray<ftype[3][8]> box;
  for (int d = 0; d < 3; ++d) {
    box(d, 0) = xc(index, 0, 0, 0, d);
    box(d, 1) = xc(index, 0, 0, p, d);
    box(d, 2) = xc(index, 0, p, p, d);
    box(d, 3) = xc(index, 0, p, 0, d);
    box(d, 4) = xc(index, p, 0, 0, d);
    box(d, 5) = xc(index, p, 0, p, d);
    box(d, 6) = xc(index, p, p, p, d);
    box(d, 7) = xc(index, p, p, 0, d);
  }
  return box;
}

template <int p>
KOKKOS_FUNCTION LocalArray<ftype[3][4]>
face_vertex_coordinates(int index, const const_face_vector_view<p>& xc)
{
  LocalArray<ftype[3][4]> base_box;
  for (int d = 0; d < 3; ++d) {
    base_box(d, 0) = xc(index, 0, 0, d);
    base_box(d, 1) = xc(index, 0, p, d);
    base_box(d, 2) = xc(index, p, p, d);
    base_box(d, 3) = xc(index, p, 0, d);
  }
  return base_box;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
