// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "matrix_free/GreenGaussBoundaryClosure.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/StkGradientFixture.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_simd/Simd.hpp"
#include "stk_topology/topology.hpp"

#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

class GradientBoundaryFixture : public GradientFixture
{
protected:
  static constexpr int order = 1;
  GradientBoundaryFixture()
    : GradientFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh(), meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 3),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 3),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 3),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 3),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(),
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      grad_bc_faces(face_node_map<order>(
        mesh(), meta.get_topology_root_part(stk::topology::QUAD_4))),
      grad_bc_offsets(face_offsets<order>(
        mesh(), meta.get_topology_root_part(stk::topology::QUAD_4), elid))
  {
    owned_lhs.putScalar(0.);
    owned_rhs.putScalar(0.);
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;
  Tpetra::MultiVector<> owned_and_shared_lhs;
  Tpetra::MultiVector<> owned_and_shared_rhs;

  const const_entity_row_view_type elid;
  const const_face_mesh_index_view<order> grad_bc_faces;
  const const_face_offset_view<order> grad_bc_offsets;

  const double some_value = -2.3;
  static constexpr int nx = 4;
  static constexpr double scale = 1;
};

TEST_F(GradientBoundaryFixture, bc_residual)
{
  auto q_face = face_scalar_view<order>("q_face", grad_bc_faces.extent_int(0));
  Kokkos::deep_copy(q_face, some_value);

  auto face_coords =
    face_vector_view<order>("face_coords", grad_bc_faces.extent_int(0));
  field_gather<order>(
    grad_bc_faces,
    stk::mesh::get_updated_ngp_field<double>(*meta.coordinate_field()),
    face_coords);
  auto exposed_areas = geom::exposed_areas<order>(face_coords);

  owned_and_shared_rhs.putScalar(0.);
  gradient_boundary_closure<order>(
    grad_bc_offsets, q_face, exposed_areas,
    owned_and_shared_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  auto view_h = owned_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    maxval = std::max(maxval, std::abs(view_h(k, 0)));
  }
  ASSERT_DOUBLE_EQ(maxval, std::abs(some_value / (scale * nx * scale * nx)));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
