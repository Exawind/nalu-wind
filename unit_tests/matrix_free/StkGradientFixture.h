
#ifndef STK_GRADIENT_FIXTURE_H
#define STK_GRADIENT_FIXTURE_H

#include "Tpetra_Map_decl.hpp"
#include "gtest/gtest.h"
#include "mpi.h"

#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_topology/topology.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/CoordinateMapping.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/Hex27Fixture.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/HexFixture.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class GradientFixture : public ::testing::Test
{
protected:
  using gid_type = typename Tpetra::Map<>::global_ordinal_type;
  static constexpr int order = 1;
  GradientFixture(int nx, double scale);
  stk::mesh::Field<double, stk::mesh::Cartesian3d>& coordinate_field()
  {
    return *meta.get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
      stk::topology::NODE_RANK, "coordinates");
  }
  stk::mesh::NgpMesh& mesh() { return bulk.get_updated_ngp_mesh(); }

  stk::mesh::Selector active() { return meta.universal_part(); }
  stk::mesh::Selector side()
  {
    return meta.get_topology_root_part(stk::topology::QUAD_4);
  }

  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  stk::io::StkMeshIoBroker io;
  stk::mesh::Field<double>& q_field;
  stk::mesh::Field<double>& dqdx_field;
  stk::mesh::Field<double>& dqdx_tmp_field;
  stk::mesh::Field<double>& dqdx_exact_field;
  stk::mesh::Field<gid_type>& gid_field;
  stk::mesh::NgpField<gid_type> gid_field_ngp;
};
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
