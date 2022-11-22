// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumDiagonal.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/KokkosViewTypes.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_simd/Simd.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace test_momentum_diag {

static constexpr int order = 1;
static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
static constexpr int num_elems = 1;

Teuchos::RCP<const Tpetra::Map<>>
make_map()
{
  return Teuchos::make_rcp<Tpetra::Map<>>(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    num_elems * nodes_per_elem, 1,
    Teuchos::make_rcp<Teuchos::MpiComm<int>>(MPI_COMM_WORLD));
}

template <int p>
void
set_aux_fields(elem_offset_view<p> offsets, vector_view<p> coordinates)
{
  constexpr auto nodes = GLL<p>::nodes;
  Kokkos::parallel_for(
    num_elems, KOKKOS_LAMBDA(int index) {
      for (int k = 0; k < p + 1; ++k) {
        const auto cz = nodes[k];
        for (int j = 0; j < p + 1; ++j) {
          const auto cy = nodes[j];
          for (int i = 0; i < p + 1; ++i) {
            const auto cx = nodes[i];
            coordinates(index, k, j, i, 0) = cx;
            coordinates(index, k, j, i, 1) = cy;
            coordinates(index, k, j, i, 2) = cz;
            offsets(index, 0, k, j, i) = 1 + index * simd_len * nodes_per_elem +
                                         k * (order + 1) * (order + 1) +
                                         j * (order + 1) + i;
          }
        }
      }
    });
}

} // namespace test_momentum_diag
class MomentumDiagonalFixture : public ::testing::Test
{
public:
  MomentumDiagonalFixture()
  {
    test_momentum_diag::set_aux_fields<order>(offsets, xc);

    scalar_view<order> pressure{"pressure", num_elems};
    Kokkos::deep_copy(pressure, 0);

    scalar_view<order> density{"density", num_elems};
    Kokkos::deep_copy(density, 1.0);
    Kokkos::deep_copy(um1, 1.0);
    Kokkos::deep_copy(up0, 1.0);
    Kokkos::deep_copy(up1, 1.0);
    Kokkos::deep_copy(visc, 1.0);
    Kokkos::deep_copy(gp, 0.0);

    auto areas = geom::linear_areas<order>(xc);
    metric = geom::diffusion_metric<order>(xc);
    vol = geom::volume_metric<order>(xc);

    geom::linear_advection_metric<order>(
      1., areas, metric, density, up1, gp, pressure, mdot);
  }
  static constexpr int order = 1;
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int num_elems = 1;

  Kokkos::Array<double, 3> gammas{{1, -1, 0}};
  vector_view<order> xc{"coords", num_elems};
  vector_view<order> um1{"um1", num_elems};
  vector_view<order> up0{"up0", num_elems};
  vector_view<order> up1{"up1", num_elems};
  scalar_view<order> visc{"visc", num_elems};
  vector_view<order> gp{"gp", num_elems};

  elem_offset_view<order> offsets{"offsets", num_elems};
  scalar_view<order> vol{"vol", num_elems};
  scs_scalar_view<order> mdot{"mdot", num_elems};
  scs_vector_view<order> metric{"metric", num_elems};
  Tpetra::MultiVector<> delta{test_momentum_diag::make_map(), 3};
  Tpetra::MultiVector<> rhs{test_momentum_diag::make_map(), 3};
};

TEST_F(MomentumDiagonalFixture, diagonal_executes)
{
  rhs.putScalar(0.);
  advdiff_diagonal<order>(
    1., offsets, vol, mdot, metric,
    rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
