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
#include <stk_unit_test_utils/stk_mesh_fixtures/HexFixture.hpp>
#include <stk_mesh/base/SkinMesh.hpp>

#include <master_element/QuadratureRule.h>
#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedElementIO.h>

#include <NaluEnv.h>

#include <memory>
#include <random>

#include "UnitTestUtils.h"

namespace {

typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<int> ScalarIntFieldType;
typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;

size_t
count_nodes(
  const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
{
  size_t nodeCount = 0u;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, selector)) {
    nodeCount += ib->size();
  }
  return nodeCount;
}

} // namespace

class PromoteElementHexTest : public ::testing::Test
{
protected:
  PromoteElementHexTest() : comm(MPI_COMM_WORLD), nDim(3){};

  void init(int nx, int ny, int nz, int in_polyOrder)
  {
    auto aura = stk::mesh::BulkData::NO_AUTO_AURA;
    fixture =
      std::make_unique<stk::mesh::fixtures::HexFixture>(comm, nx, ny, nz, aura);
    meta = &fixture->m_meta;
    bulk = &fixture->m_bulk_data;
    surfSupPart = nullptr;
    surfSubPart = nullptr;
    topo = stk::topology::HEX_8;
    hexPart = fixture->m_elem_parts[0];
    ThrowRequire(hexPart != nullptr);
    coordField =
      &meta->declare_field<VectorFieldType>(stk::topology::NODE_RANK, "coords");
    intField = &meta->declare_field<ScalarIntFieldType>(
      stk::topology::NODE_RANK, "integer field");

    poly_order = in_polyOrder;

    surfSupPart = &meta->declare_part("surface_1", stk::topology::FACE_RANK);
    surfSubPart = &meta->declare_part_with_topology(
      "surface_1_hex8_quad4", stk::topology::QUAD_4);

    meta->declare_part_subset(*surfSupPart, *surfSubPart);
    edgePart = &meta->declare_part("edge_part", stk::topology::EDGE_RANK);
    facePart = &meta->declare_part("face_part", stk::topology::FACE_RANK);
    baseParts = {hexPart, surfSupPart};

    setup_promotion();

    stk::mesh::put_field_on_entire_mesh(*coordField, nDim);
    stk::mesh::put_field_on_entire_mesh(*intField);

    fixture->m_meta.commit();
    fixture->generate_mesh(stk::mesh::fixtures::FixedCartesianCoordinateMapping(
      nx, ny, nz, nx, ny, nz));
    stk::mesh::PartVector surfParts = {surfSubPart};
    stk::mesh::skin_mesh(*bulk, surfParts);
  }

  void setup_promotion()
  {
    // declare super parts mirroring the orginal parts
    sierra::nalu::HexNElementDescription desc(poly_order);
    const auto superName =
      sierra::nalu::super_element_part_name(hexPart->name());
    topo = stk::create_superelement_topology(
      static_cast<unsigned>(desc.nodesPerElement));
    const stk::mesh::Part* superPart =
      &meta->declare_part_with_topology(superName, topo);
    superParts.push_back(superPart);

    stk::mesh::Part* superSuperPart = &meta->declare_part(
      sierra::nalu::super_element_part_name(surfSupPart->name()),
      stk::topology::FACE_RANK);

    const auto sidePartName =
      sierra::nalu::super_subset_part_name(surfSubPart->name());
    auto sideTopo =
      stk::create_superface_topology(static_cast<unsigned>(desc.nodesPerSide));
    stk::mesh::Part* superSidePart =
      &meta->declare_part_with_topology(sidePartName, sideTopo);
    meta->declare_part_subset(*superSuperPart, *superSidePart);
    superParts.push_back(superSuperPart);
  }

  void promote_mesh()
  {
    auto gllNodes =
      sierra::nalu::gauss_lobatto_legendre_rule(poly_order + 1).first;
    sierra::nalu::promotion::create_tensor_product_hex_elements(
      gllNodes, *bulk, *coordField, baseParts);
  }

  void output_mesh()
  {
    const stk::mesh::PartVector& outParts = {hexPart};
    std::string fileName = "hv2.e";

    io = std::make_unique<sierra::nalu::PromotedElementIO>(
      poly_order, *meta, *bulk, outParts, fileName, *coordField);
    io->write_database_data(0.0);
  }

  size_t expected_node_count(size_t originalNodeCount)
  {
    size_t expectedNodeCount = std::pow(
      poly_order * (static_cast<int>(std::cbrt(originalNodeCount + 1)) - 1) + 1,
      3);
    return expectedNodeCount;
  }

  stk::ParallelMachine comm;
  unsigned nDim;
  std::unique_ptr<stk::mesh::fixtures::HexFixture> fixture;
  stk::mesh::MetaData* meta;
  stk::mesh::BulkData* bulk;
  unsigned poly_order;
  stk::mesh::Part* hexPart;
  stk::mesh::Part* surfSupPart;
  stk::mesh::Part* surfSubPart;
  stk::topology topo;
  stk::mesh::PartVector baseParts;
  stk::mesh::ConstPartVector superParts;
  stk::mesh::Part* edgePart;
  stk::mesh::Part* facePart;
  std::unique_ptr<sierra::nalu::PromotedElementIO> io;
  VectorFieldType* coordField;
  ScalarIntFieldType* intField;
};

TEST_F(PromoteElementHexTest, node_count)
{
  int polyOrder = 7;

  int nprocs = stk::parallel_machine_size(MPI_COMM_WORLD);
  int nprocx = std::cbrt(nprocs + 0.5);
  if (nprocx * nprocx * nprocx != nprocs) {
    return;
  }

  init(nprocx, nprocx, nprocx, polyOrder);
  size_t originalNodeCount = ::count_nodes(*bulk, meta->universal_part());

  stk::mesh::PartVector supElemParts;
  stk::mesh::PartVector supSideParts;

  promote_mesh();
  EXPECT_EQ(
    expected_node_count(originalNodeCount),
    ::count_nodes(*bulk, meta->universal_part()));

  bool outputMesh = false;
  if (outputMesh) {
    EXPECT_NO_THROW(output_mesh());
  }
}

TEST_F(PromoteElementHexTest, node_sharing)
{
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 2) {
    return;
  }

  int polyOrder = 2;
  init(2, 1, 1, polyOrder);

  promote_mesh();
  ThrowRequire(!bulk->in_modifiable_state());

  stk::mesh::EntityIdVector sharedNodeIds = {2, 5, 8, 11, 21, 22, 23, 24, 33};

  for (auto id : sharedNodeIds) {
    auto newSharedNode = bulk->get_entity(stk::topology::NODE_RANK, id);
    *stk::mesh::field_data(*intField, newSharedNode) =
      bulk->parallel_rank() + 1;
    if (bulk->parallel_size() > 1) {
      stk::mesh::parallel_sum(*bulk, {intField});
    }
    EXPECT_EQ(*stk::mesh::field_data(*intField, newSharedNode), 3);
  }
}
