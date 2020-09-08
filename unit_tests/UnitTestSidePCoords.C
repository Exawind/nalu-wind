#include <gtest/gtest.h>

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

#include <master_element/MasterElementFactory.h>
#include <nalu_make_unique.h>
#include <NaluEnv.h>

#include <memory>
#include <tuple>
#include <random>
#include <limits>

#include "UnitTestUtils.h"

namespace
{
  void check_elem_to_side_coords(stk::topology topo)
  {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) > 1) { return; }

    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<double> perturb(0.125, 0.25);
    std::vector<double> sideIsoPoint = { perturb(rng), perturb(rng) };

    int dim = topo.dimension();
    stk::mesh::MetaData meta(topo.dimension());
    stk::mesh::BulkData bulk(meta, MPI_COMM_WORLD);

    auto elem = unit_test_utils::create_one_reference_element(bulk, topo);
    const stk::mesh::Entity* elem_node_rels = bulk.begin_nodes(elem);
    auto* meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

    using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;
    const VectorFieldType& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());

    // perturb the reference element

    for (int n = 0; n < meSCS->nodesPerElement_; ++n) {
      stk::mesh::Entity elem_node = elem_node_rels[n];
      double* coords =  stk::mesh::field_data(coordField, elem_node);
      for (int d = 0; d < dim; ++d) {
        coords[d] += perturb(rng);
      }
    }

    const auto& face_buckets = bulk.get_buckets(meta.side_rank(), meta.universal_part());
    for (const auto* ib : face_buckets) {
      const auto& b = *ib;

      auto* meSide = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
      std::vector<double> sideInterpWeights(meSide->num_integration_points() * meSide->nodesPerElement_, 0.0);
      meSide->shape_fcn(sideInterpWeights.data());

      for (size_t k = 0; k < b.size(); ++k) {
        auto face = b[k];
        const auto* face_node_rels = bulk.begin_nodes(face);

        std::vector<double> faceCoords(dim * meSide->nodesPerElement_);
        for (int d = 0; d < dim; ++d) {
          for (int n = 0; n < meSide->nodesPerElement_; ++n) {
            stk::mesh::Entity face_node = face_node_rels[n];
            double* coords =  stk::mesh::field_data(coordField, face_node);
            faceCoords[d*meSide->nodesPerElement_ + n] = coords[d];
          }
        }

        std::vector<double> elemCoords(dim * meSCS->nodesPerElement_);
        for (int d = 0; d < dim; ++d) {
          for (int n = 0; n < meSCS->nodesPerElement_; ++n) {
            stk::mesh::Entity elem_node = elem_node_rels[n];
            double* coords =  stk::mesh::field_data(coordField, elem_node);
            elemCoords[d*meSCS->nodesPerElement_ + n] = coords[d];
          }
        }

        std::vector<double> sideElemCoords(dim, 0.0);
        const int side_ordinal = bulk.begin_element_ordinals(face)[0];
        meSCS->sidePcoords_to_elemPcoords(side_ordinal, 1, sideIsoPoint.data(), sideElemCoords.data());

        std::vector<double> faceInterpCoords(dim, 0.0);
        meSide->interpolatePoint(dim, sideIsoPoint.data(), faceCoords.data(), faceInterpCoords.data());

        std::vector<double> isoFacePoint(dim , 0.0);
        meSCS->isInElement(elemCoords.data(), faceInterpCoords.data(), isoFacePoint.data());

        // quad/hex isInElement uses a different isoparameteric range
        double isoFac = (topo == stk::topology::HEX_8 || topo == stk::topology::QUAD_4_2D) ? 0.5 : 1.0;
        for (int d = 0; d < dim; ++d) {
          isoFacePoint[d] *= isoFac;
        }

        for (int d = 0; d < dim; ++d) {
          EXPECT_NEAR(isoFacePoint[d], sideElemCoords[d], 1.0e-8);
        }
      }
    }
  }

} // namespace

#define TEST_ALL_TOPOS_NO_PYR(x,y) \
    TEST(x, tri3##_##y)   { y(stk::topology::TRI_3_2D); } \
    TEST(x, quad4##_##y)  { y(stk::topology::QUAD_4_2D); } \
    TEST(x, quad9##_##y)  { y(stk::topology::QUAD_9_2D); } \
    TEST(x, tet4##_##y)   { y(stk::topology::TET_4); } \
    TEST(x, wedge6##_##y) { y(stk::topology::WEDGE_6); } \
    TEST(x, hex8##_##y)   { y(stk::topology::HEX_8); } \
    TEST(x, hex27##_##y)  { y(stk::topology::HEX_27); }

TEST_ALL_TOPOS_NO_PYR(SidePCoords, check_elem_to_side_coords)

