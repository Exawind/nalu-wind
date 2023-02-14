#include <gtest/gtest.h>
#include <limits>

#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>

#ifdef NALU_HAS_MATRIXFREE
#include <matrix_free/LobattoQuadratureRule.h>
#endif
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>
#include <element_promotion/HexNElementDescription.h>

#include <NaluEnv.h>
#include <BucketLoop.h>

#include <memory>
#include <tuple>
#include <random>

#include "UnitTestUtils.h"

namespace {
void
fill_and_promote_hex_mesh(
  const std::string& meshSpec, stk::mesh::BulkData& bulk, int polyOrder)
{
  stk::io::StkMeshIoBroker io(bulk.parallel());
  io.set_bulk_data(bulk);
  io.add_mesh_database(meshSpec, stk::io::READ_MESH);
  io.create_input_mesh();

  stk::mesh::MetaData& meta = bulk.mesh_meta_data();
  stk::mesh::Part* blockPart = meta.get_part("block_1");
  stk::mesh::Part* surfPart =
    &meta.declare_part_with_topology("surface_1", stk::topology::QUAD_4);

  auto elemDesc = sierra::nalu::HexNElementDescription(polyOrder);

  const std::string superName =
    sierra::nalu::super_element_part_name("block_1");
  stk::topology topo = stk::create_superelement_topology(
    static_cast<unsigned>(elemDesc.nodesPerElement));
  meta.declare_part_with_topology(superName, topo);

  stk::mesh::Part* superSuperPart = &meta.declare_part(
    sierra::nalu::super_element_part_name("surface_1"),
    stk::topology::FACE_RANK);

  const auto sidePartName = sierra::nalu::super_subset_part_name("surface_1");
  auto sideTopo = stk::create_superface_topology(
    static_cast<unsigned>(elemDesc.nodesPerSide));
  stk::mesh::Part* superSidePart =
    &meta.declare_part_with_topology(sidePartName, sideTopo);
  meta.declare_part_subset(*superSuperPart, *superSidePart);

  meta.declare_part("edge_part", stk::topology::EDGE_RANK);
  meta.declare_part("face_part", stk::topology::FACE_RANK);

  io.populate_bulk_data();
  stk::mesh::create_exposed_block_boundary_sides(bulk, *blockPart, {surfPart});

  VectorFieldType* coords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::PartVector baseParts = {blockPart, surfPart};
  std::vector<double> nodes(polyOrder + 1);
  for (size_t j = 0; j < polyOrder + 1; ++j) {
    nodes[j] = -1 + 2. / (polyOrder)*j;
  }
  sierra::nalu::promotion::create_tensor_product_hex_elements(
    nodes, bulk, *coords, baseParts);
}

void
dump_promoted_mesh_file(stk::mesh::BulkData& bulk, int polyOrder)
{
  const auto& meta = bulk.mesh_meta_data();
  const stk::mesh::PartVector& outParts = meta.get_mesh_parts();
  std::string fileName = "out.e";

  auto desc = sierra::nalu::HexNElementDescription(polyOrder);
  VectorFieldType* coordField =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  sierra::nalu::PromotedElementIO io(
    polyOrder, meta, bulk, outParts, fileName, *coordField);
  io.write_database_data(0.0);
}

} // namespace
TEST(SingleHexPromotion, coords_p2)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) {
    return;
  }

  // Hex 27 standard node locations for a [0,1]^3 element, with the center node
  // moved from index 20 to index 26.
  std::vector<std::vector<double>> expectedCoords = {
    {+0.0, +0.0, +0.0}, {+1.0, +0.0, +0.0}, {+1.0, +1.0, +0.0},
    {+0.0, +1.0, +0.0}, {+0.0, +0.0, +1.0}, {+1.0, +0.0, +1.0},
    {+1.0, +1.0, +1.0}, {+0.0, +1.0, +1.0}, {+0.5, +0.0, +0.0},
    {+1.0, +0.5, +0.0}, {+0.5, +1.0, +0.0}, {+0.0, +0.5, +0.0},
    {+0.0, +0.0, +0.5}, {+1.0, +0.0, +0.5}, {+1.0, +1.0, +0.5},
    {+0.0, +1.0, +0.5}, {+0.5, +0.0, +1.0}, {+1.0, +0.5, +1.0},
    {+0.5, +1.0, +1.0}, {+0.0, +0.5, +1.0}, {+0.5, +0.5, +0.0},
    {+0.5, +0.5, +1.0}, {+0.0, +0.5, +0.5}, {+1.0, +0.5, +0.5},
    {+0.5, +0.0, +0.5}, {+0.5, +1.0, +0.5}, {+0.5, +0.5, +0.5}};

  int dim = 3;
  int polynomialOrder = 2;

  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(dim);
  meshBuilder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
  auto bulk = meshBuilder.create();
  auto& meta = bulk->mesh_meta_data();

  std::string singleElemMeshSpec = "generated:1x1x1";
  fill_and_promote_hex_mesh(singleElemMeshSpec, *bulk, polynomialOrder);
  const stk::mesh::PartVector promotedElemParts =
    sierra::nalu::only_super_elem_parts(meta.get_parts());
  const stk::mesh::Selector promotedElemSelector =
    stk::mesh::selectUnion(promotedElemParts);
  const stk::mesh::BucketVector& buckets =
    bulk->get_buckets(stk::topology::ELEM_RANK, promotedElemSelector);

  stk::mesh::EntityVector elems;
  stk::mesh::get_selected_entities(promotedElemSelector, buckets, elems);
  ASSERT_EQ(elems.size(), 1u);

  VectorFieldType* coordField =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  for (stk::mesh::Entity elem : elems) {
    const stk::mesh::Entity* elemNodeRelations = bulk->begin_nodes(elem);
    for (unsigned k = 0; k < bulk->num_nodes(elem); ++k) {
      const stk::mesh::Entity node = elemNodeRelations[k];
      const double* stkCoordsForNodeK =
        stk::mesh::field_data(*coordField, node);
      for (int d = 0; d < dim; ++d) {
        EXPECT_NEAR(stkCoordsForNodeK[d], expectedCoords[k][d], 1.0e-14);
      }
    }
  }

  bool doOutput = false;
  if (doOutput) {
    dump_promoted_mesh_file(*bulk, polynomialOrder);
  }
}
