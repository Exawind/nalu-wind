// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/SparsifiedEdgeCrsMatrix.h"
#include "matrix_free/SparsifiedEdgeLaplacian.h"

#include "Tpetra_Assembly_Helpers.hpp"

#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionInterior.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraMap.h"
#include "StkConductionFixture.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stk_mesh/base/GetNgpField.hpp>
#include "stk_mesh/base/GetNgpMesh.hpp"

#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace test_solution_update {
static constexpr Kokkos::Array<double, 3> gammas = {{0, 0, 0}};
}

class SparsifiedEdgeFixture : public ::ConductionFixture
{
protected:
  SparsifiedEdgeFixture()
    : ConductionFixture(nx, scale),
      linsys(
        stk::mesh::get_updated_ngp_mesh(bulk),
        meta.universal_part(),
        gid_field_ngp),
      offsets(
        create_offset_map<order>(
          mesh, meta.universal_part(), linsys.stk_lid_to_tpetra_lid))
  {
  }
  StkToTpetraMaps linsys;
  const_elem_offset_view<order> offsets;
  static constexpr int nx = 32;
  static constexpr double scale = 32;
};

TEST_F(SparsifiedEdgeFixture, create_matrix)
{
  create_matrix<order>(linsys.owned, linsys.owned_and_shared, offsets);
  ASSERT_NO_THROW(
    create_matrix<order>(linsys.owned, linsys.owned_and_shared, offsets));
}

TEST_F(SparsifiedEdgeFixture, fill_works)
{
  auto xc = vector_view<order>("coords", offsets.extent_int(0));
  auto coord_ngp = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  auto conn = stk_connectivity_map<order>(mesh, meta.universal_part());
  field_gather<order>(conn, coord_ngp, xc);
  auto mat =
    create_matrix<order>(linsys.owned, linsys.owned_and_shared, offsets);

  ASSERT_NO_THROW(
    assemble_sparsified_edge_laplacian<order>(
      offsets, xc, mat->getLocalMatrix()));
  ASSERT_NO_THROW(mat->fillComplete());
}

TEST_F(SparsifiedEdgeFixture, dump)
{
  auto xc = vector_view<order>("coords", offsets.extent_int(0));
  auto coord_ngp = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  auto conn = stk_connectivity_map<order>(mesh, meta.universal_part());
  field_gather<order>(conn, coord_ngp, xc);
  auto mat =
    create_matrix<order>(linsys.owned, linsys.owned_and_shared, offsets);
  assemble_sparsified_edge_laplacian<order>(
    offsets, xc, linsys.owned_and_shared.getLocalMap(), *mat);
  mat->fillComplete();

  Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<>>::writeSparseFile(
    "laplacian", *mat);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
