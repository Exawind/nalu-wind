/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "ngp_algorithms/GeometryInteriorAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/WallFuncGeometryAlg.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "utils/StkHelpers.h"

TEST_F(TestKernelHex8Mesh, NGP_geometry_interior)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // zero out fields
  stk::mesh::field_fill(0.0, *dnvField_);
  stk::mesh::field_fill(0.0, *elementVolume_);
  stk::mesh::field_fill(0.0, *edgeAreaVec_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Force computation of edge area vector
  helperObjs.realm.realmUsesEdges_ = true;

  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);

  geomAlgDriver.register_elem_algorithm<sierra::nalu::GeometryInteriorAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "geometry");

  geomAlgDriver.execute();

  const double tol = 1.0e-16;
  stk::mesh::Selector sel = meta_.universal_part();

  // Dual volume check
  {
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int counter = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dVol = stk::mesh::field_data(*dnvField_, node);
        EXPECT_NEAR(0.125, dVol[0], tol);
        counter++;
      }
    EXPECT_EQ(counter, 8);
  }

  // Element volume check
  {
    const auto& bkts = bulk_.get_buckets(stk::topology::ELEM_RANK, sel);

    int counter = 0;
    for (const auto* b: bkts)
      for (const auto elem: *b) {
        const double* dVol = stk::mesh::field_data(*elementVolume_, elem);
        EXPECT_NEAR(1.0, dVol[0], tol);
        counter++;
      }
    EXPECT_EQ(counter, 1);
  }

  // Edge area vector check
  {
    const auto& bkts = bulk_.get_buckets(stk::topology::EDGE_RANK, sel);
    const double aMagSqrGold = 0.25 * 0.25;

    int counter = 0;
    for (const auto* b: bkts)
      for (const auto edge: *b) {
        const double* areaVec = stk::mesh::field_data(*edgeAreaVec_, edge);
        double aMagSqr = 0.0;
        for (int i=0; i < 3; i++)
          aMagSqr += areaVec[i] * areaVec[i];

        EXPECT_NEAR(aMagSqrGold, aMagSqr, tol);
        counter++;
      }
    EXPECT_EQ(counter, 12);
  }
}

TEST_F(TestKernelHex8Mesh, NGP_geometry_bndry)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // zero out fields
  stk::mesh::field_fill(0.0, *exposedAreaVec_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  auto* part = meta_.get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
  geomAlgDriver.register_face_algorithm<sierra::nalu::GeometryBoundaryAlg>(
    sierra::nalu::WALL, surfPart, "geometry");

  geomAlgDriver.execute();

  // Exposed area vector check
  {
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_.get_buckets(stk::topology::FACE_RANK, sel);
    const double aMagSqrGold = 0.25 * 0.25;

    const double tol = 1.0e-16;
    for (const auto* b: bkts)
      for (const auto face: *b) {
        const double* areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
        for (int ip=0; ip < sierra::nalu::AlgTraitsQuad4::numFaceIp_; ++ip) {
          double aMagSqr = 0.0;
          for (int i=0; i < 3; i++) {
            const double av = areaVec[ip * sierra::nalu::AlgTraitsQuad4::nDim_ + i];
            aMagSqr += av * av;
          }

          EXPECT_NEAR(aMagSqrGold, aMagSqr, tol);
        }
      }
  }
}

TEST_F(KsgsKernelHex8Mesh, NGP_geometry_wall_func)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  LowMachKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets);

  // zero out fields
  stk::mesh::field_fill(0.0, *wallNormDist_);
  stk::mesh::field_fill(0.0, *wallArea_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  auto* part = meta_.get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
  geomAlgDriver.register_wall_func_algorithm<sierra::nalu::WallFuncGeometryAlg>(
    sierra::nalu::WALL, surfPart,
    sierra::nalu::get_elem_topo(helperObjs.realm, *surfPart), "geometry");

  geomAlgDriver.execute();

  // wall distance and area check
  {
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
    const double wdistGold = 0.25;
    const double wAreaGold = 0.25;

    const double tol = 1.0e-16;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* wdist = stk::mesh::field_data(*wallNormDist_, node);
        const double* warea = stk::mesh::field_data(*wallArea_, node);
        EXPECT_NEAR(wdistGold, wdist[0], tol);
        EXPECT_NEAR(wAreaGold, warea[0], tol);
      }
  }
}
