#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include "UnitTestUtils.h"

namespace {

unsigned
count_locally_owned_elems(const stk::mesh::BulkData& bulk)
{
  return stk::mesh::count_selected_entities(
    bulk.mesh_meta_data().locally_owned_part(),
    bulk.buckets(stk::topology::ELEM_RANK));
}

void
verify_elems_are_unit_cubes(const stk::mesh::BulkData& bulk)
{
  typedef stk::mesh::Field<double> CoordFieldType;
  CoordFieldType* coordField = bulk.mesh_meta_data().get_field<double>(
    stk::topology::NODE_RANK, "coordinates");
  EXPECT_TRUE(coordField != nullptr);

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);

  const double tolerance = std::numeric_limits<double>::epsilon();

  for (stk::mesh::Entity elem : elems) {
    double minX = 999.99, minY = 999.99, minZ = 999.99;
    double maxX = 0, maxY = 0, maxZ = 0;
    const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
    unsigned numNodes = bulk.num_nodes(elem);
    for (unsigned i = 0; i < numNodes; ++i) {
      double* nodeCoords = stk::mesh::field_data(*coordField, nodes[i]);
      minX = std::min(minX, nodeCoords[0]);
      minY = std::min(minY, nodeCoords[1]);
      minZ = std::min(minZ, nodeCoords[2]);
      maxX = std::max(maxX, nodeCoords[0]);
      maxY = std::max(maxY, nodeCoords[1]);
      maxZ = std::max(maxZ, nodeCoords[2]);
    }

    EXPECT_NEAR(1.0, (maxX - minX), tolerance);
    EXPECT_NEAR(1.0, (maxY - minY), tolerance);
    EXPECT_NEAR(1.0, (maxZ - minZ), tolerance);
  }
}

} // namespace

TEST(Basic, CheckCoords1Elem)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;

  unsigned spatialDimension = 3;
  stk::mesh::MeshBuilder meshBuilder(comm);
  meshBuilder.set_spatial_dimension(spatialDimension);
  auto bulk = meshBuilder.create();
  bulk->mesh_meta_data().use_simple_fields();

  unit_test_utils::fill_mesh_1_elem_per_proc_hex8(*bulk);

  EXPECT_EQ(1u, count_locally_owned_elems(*bulk));

  verify_elems_are_unit_cubes(*bulk);
}
