#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/Hex27CVFEM.h>
#include <master_element/Quad93DCVFEM.h>

#include "UnitTestUtils.h"

namespace {

void check_Hex27_creation(const stk::mesh::BulkData& bulk)
{
  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  stk::topology hex27 = stk::topology::HEX_27;
  stk::topology quad9 = stk::topology::QUADRILATERAL_9;

  std::vector<std::vector<unsigned>> faceNodeOrdinals(6);
  faceNodeOrdinals[0] = {0, 1, 5, 4,  8, 13, 16, 12, 25};
  faceNodeOrdinals[1] = {1, 2, 6, 5,  9, 14, 17, 13, 24};
  faceNodeOrdinals[2] = {2, 3, 7, 6, 10, 15, 18, 14, 26};
  faceNodeOrdinals[3] = {0, 4, 7, 3, 12, 19, 15, 11, 23};
  faceNodeOrdinals[4] = {0, 3, 2, 1, 11, 10,  9,  8, 21};
  faceNodeOrdinals[5] = {4, 5, 6, 7, 16, 17, 18, 19, 22};

  for(stk::mesh::Entity elem : elems) {
    EXPECT_EQ(hex27, bulk.bucket(elem).topology());
    const stk::mesh::ConnectivityOrdinal* ords = bulk.begin_face_ordinals(elem);
    const stk::mesh::Entity* faces = bulk.begin_faces(elem);
    const stk::mesh::Entity* elem_node_rels = bulk.begin_nodes(elem);

    for (unsigned j = 0; j < bulk.num_faces(elem); ++j) {
      stk::mesh::Entity face = faces[ords[j]];
      EXPECT_EQ(quad9, bulk.bucket(face).topology());
      const stk::mesh::Entity* face_node_rels = bulk.begin_nodes(face);

      EXPECT_EQ(quad9.num_nodes(), bulk.num_nodes(face));
      for (unsigned i = 0; i < bulk.num_nodes(face); ++i) {
        stk::mesh::EntityId nodeId = bulk.identifier(face_node_rels[i]);

        stk::mesh::Entity expectedNode = elem_node_rels[faceNodeOrdinals[ords[j]][i]];
        stk::mesh::EntityId expectedNodeId = bulk.identifier(expectedNode);
        EXPECT_EQ(expectedNodeId, nodeId);
      }
    }
  }
}

void check_Hex27_face_ip_node_ordering(const stk::mesh::BulkData& bulk)
{
  stk::topology hex27 = stk::topology::HEX_27;
  sierra::nalu::Hex27SCS hexSCS;
  sierra::nalu::Quad93DSCS quadSCS;

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);
  for(stk::mesh::Entity elem : elems) {
    const stk::mesh::ConnectivityOrdinal* ords = bulk.begin_face_ordinals(elem);
    const stk::mesh::Entity* faces = bulk.begin_faces(elem);
    const stk::mesh::Entity* elem_node_rels = bulk.begin_nodes(elem);

    EXPECT_EQ(hex27.num_faces(), bulk.num_faces(elem));
    for (unsigned j = 0; j < bulk.num_faces(elem); ++j) {
      const stk::mesh::ConnectivityOrdinal ordinal = ords[j];
      stk::mesh::Entity face = faces[ordinal];
      const stk::mesh::Entity* face_node_rels = bulk.begin_nodes(face);

      const int* ipNodeMap = hexSCS.ipNodeMap(ordinal);
      const int* faceIpNodeMap = quadSCS.ipNodeMap();
      for (int ip = 0; ip < quadSCS.num_integration_points(); ++ip) {
        const int elemIpNearestNode = ipNodeMap[ip];
        const int faceIpNearestNode = faceIpNodeMap[ip];

        stk::mesh::EntityId elemNearestNodeId = bulk.identifier(elem_node_rels[elemIpNearestNode]);
        stk::mesh::EntityId faceNearestNodeId = bulk.identifier(face_node_rels[faceIpNearestNode]);
        EXPECT_EQ(elemNearestNodeId, faceNearestNodeId);
      }
    }
  }
}

}//namespace

TEST(Hex27,creation)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  if (stk::parallel_machine_size(comm) != 1) {
    return; // serial test
  }

  unsigned spatialDimension = 3;
  stk::mesh::MetaData meta(spatialDimension);
  stk::mesh::BulkData bulk(meta, comm);

  unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_27);
  check_Hex27_creation(bulk);
}

TEST(Hex27, face_node_ordering)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;
  if (stk::parallel_machine_size(comm) != 1) {
    return; // serial test
  }

  unsigned spatialDimension = 3;
  stk::mesh::MetaData meta(spatialDimension);
  stk::mesh::BulkData bulk(meta, comm);

  unit_test_utils::create_one_reference_element(bulk, stk::topology::HEX_27);
  check_Hex27_face_ip_node_ordering(bulk);
}





