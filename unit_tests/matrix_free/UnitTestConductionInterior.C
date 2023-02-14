// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Kokkos_Core.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <algorithm>
#include <stk_simd/Simd.hpp>
#include <type_traits>

#include "matrix_free/ConductionInterior.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/KokkosViewTypes.h"

#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace test_conduction {

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

} // namespace test_conduction
class ConductionResidualFixture : public ::testing::Test
{
public:
  ConductionResidualFixture()
  {
    Kokkos::deep_copy(qp1, 1.0);
    Kokkos::deep_copy(qp0, 1.0);
    Kokkos::deep_copy(qm1, 1.0);

    vector_view<order> coordinates{"coordinates", num_elems};
    test_conduction::set_aux_fields<order>(offsets, coordinates);

    scalar_view<order> alpha{"alpha", num_elems};
    Kokkos::deep_copy(alpha, 1.0);

    scalar_view<order> lambda{"lambda", num_elems};
    Kokkos::deep_copy(lambda, 1.0);

    volume_metric = geom::volume_metric<order>(alpha, coordinates);
    diffusion_metric = geom::diffusion_metric<order>(alpha, coordinates);
  }
  static constexpr int order = 1;
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int num_elems = 1;
  const Kokkos::Array<double, 3> gamma{{+1, -1, 0}};
  elem_offset_view<order> offsets{"offsets", num_elems};
  scalar_view<order> qm1{"qm1", num_elems};
  scalar_view<order> qp0{"qp0", num_elems};
  scalar_view<order> qp1{"qp1", num_elems};
  scalar_view<order> volume_metric{"volume_metric", num_elems};
  scs_vector_view<order> diffusion_metric{"diffusion_metric", num_elems};
  Tpetra::MultiVector<> delta{test_conduction::make_map(), 1};
  Tpetra::MultiVector<> rhs{test_conduction::make_map(), 1};
};

TEST_F(ConductionResidualFixture, residual_executes)
{
  decltype(rhs.getLocalViewDevice(Tpetra::Access::ReadWrite)) shared_rhs(
    "empty_rhs", 1, 1);

  rhs.putScalar(0.);
  conduction_residual<order>(
    gamma, offsets, qm1, qp0, qp1, volume_metric, diffusion_metric,
    rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}

TEST_F(ConductionResidualFixture, linearized_residual_executes)
{
  delta.putScalar(0.);
  rhs.putScalar(0.);

  rhs.putScalar(0.);
  conduction_linearized_residual<order>(
    gamma[0], offsets, volume_metric, diffusion_metric,
    delta.getLocalViewDevice(Tpetra::Access::ReadWrite),
    rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
