// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/FilterDiagonal.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "stk_simd/Simd.hpp"
#include <algorithm>
#include <type_traits>

#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "SetOffsetsAndCoordinates.h"

#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/LinearVolume.h"

#include "gtest/gtest.h"
#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;

namespace test_filter_diagonal {

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

} // namespace test_filter_diagonal
class FilterDiagonal : public ::testing::Test
{
public:
  FilterDiagonal()
  {
    vector_view<order> coordinates{"coordinates", num_elems};
    set_offsets_and_coordinates<order>(offsets, coordinates, num_elems);
    volume_metric = geom::volume_metric<order>(coordinates);
  }
  static constexpr int order = 1;
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int num_elems = 1;
  elem_offset_view<order> offsets{"offsets", num_elems};
  scalar_view<order> volume_metric{"volume_metric", num_elems};
  Tpetra::MultiVector<> rhs{test_filter_diagonal::make_map(), 3};
};

TEST_F(FilterDiagonal, diagonal_executes)
{
  node_offset_view dirichlet_offsets("empty_dirichlet", 1);
  decltype(rhs.getLocalViewDevice(Tpetra::Access::ReadWrite)) shared_rhs(
    "empty_rhs", 1, 3);

  rhs.putScalar(0.);
  filter_diagonal<order>(
    offsets, volume_metric, rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
