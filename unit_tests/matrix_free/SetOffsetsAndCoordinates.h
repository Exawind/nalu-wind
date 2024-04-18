// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Kokkos_Core.hpp>

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/LobattoQuadratureRule.h"

#include "gtest/gtest.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
void
set_offsets_and_coordinates(
  elem_offset_view<p> offsets, vector_view<p> coordinates, int nrepeated)
{
  constexpr int nodes_per_elem = (p + 1) * (p + 1) * (p + 1);
  constexpr auto nodes = GLL<p>::nodes;
  Kokkos::parallel_for(
    nrepeated, KOKKOS_LAMBDA(int index) {
      const int elem_offset = index * nodes_per_elem;

      for (int k = 0; k < p + 1; ++k) {
        const auto cz = nodes[k];
        for (int j = 0; j < p + 1; ++j) {
          const auto cy = nodes[j];
          for (int i = 0; i < p + 1; ++i) {
            const auto cx = nodes[i];
            coordinates(index, k, j, i, 0) = cx;
            coordinates(index, k, j, i, 1) = cy;
            coordinates(index, k, j, i, 2) = cz;

            const int simd_lane_0 = 0;
            offsets(index, k, j, i, simd_lane_0) =
              elem_offset + k * (p + 1) * (p + 1) + j * (p + 1) + i;
            for (int n = 1; n < simd_len; ++n) {
              offsets(index, k, j, i, n) = invalid_offset;
            }
          }
        }
      }
    });
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
