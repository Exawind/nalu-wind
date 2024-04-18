// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradientInterior.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/KokkosViewTypes.h"
#include "SetOffsetsAndCoordinates.h"

#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace test_gradient {

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

} // namespace test_gradient
class GradientResidualFixture : public ::testing::Test
{
public:
  GradientResidualFixture()
  {
    Kokkos::deep_copy(qp1, 1.0);
    Kokkos::deep_copy(dqdx, 0.0);

    vector_view<order> coordinates{"coordinates", num_elems};
    set_offsets_and_coordinates<order>(offsets, coordinates, num_elems);

    volume_metric = geom::volume_metric<order>(coordinates);
    area_metric = geom::linear_areas<order>(coordinates);
  }
  static constexpr int order = 1;
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int num_elems = 1;
  const Kokkos::Array<double, 3> gamma{{+1, -1, 0}};
  elem_offset_view<order> offsets{"offsets", num_elems};
  scalar_view<order> qp1{"qp1", num_elems};
  vector_view<order> dqdx{"dqdx", num_elems};
  scalar_view<order> volume_metric{"volume_metric", num_elems};
  scs_vector_view<order> area_metric{"area_metric", num_elems};
  Tpetra::MultiVector<> delta{test_gradient::make_map(), 3};
  Tpetra::MultiVector<> rhs{test_gradient::make_map(), 3};
};

TEST_F(GradientResidualFixture, residual_executes)
{
  rhs.putScalar(0.);
  gradient_residual<order>(
    offsets, area_metric, volume_metric, qp1, dqdx,
    rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}

TEST_F(GradientResidualFixture, linearized_residual_executes)
{
  delta.putScalar(0.);
  rhs.putScalar(0.);
  filter_linearized_residual<order>(
    offsets, volume_metric, delta.getLocalViewDevice(Tpetra::Access::ReadWrite),
    rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
