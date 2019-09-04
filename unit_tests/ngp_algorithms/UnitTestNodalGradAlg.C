/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"
#include "ngp_algorithms/UnitTestNgpAlgUtils.h"

#include "AlgTraits.h"
#include "AssembleNodalGradEdgeAlgorithm.h"
#include "AssembleNodalGradElemAlgorithm.h"
#include "AssembleNodalGradBoundaryAlgorithm.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"

#include "stk_mesh/base/CreateEdges.hpp"

TEST_F(SSTKernelHex8Mesh, ngp_nodal_grad_edge)
{
  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradEdge
  // sierra::nalu::AssembleNodalGradEdgeAlgorithm edgeAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_);
  // edgeAlg.execute();
  sierra::nalu::ScalarNodalGradEdgeAlg edgeAlg(
    helperObjs.realm, partVec_[0], tke_, dkdx_);
  edgeAlg.execute();

  {
    std::vector<double> expectedValues = {
      2, 2, 2, -2, 6, 6,
      6, -2, 6, -6, -6, 10,
      6, 6, -2, -6, 10, -6,
      10, -6, -6, -10, -10, -10
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, ngp_nodal_grad_elem)
{
  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradElem
  // sierra::nalu::AssembleNodalGradElemAlgorithm elemAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_);
  // elemAlg.execute();
  sierra::nalu::ScalarNodalGradElemAlg<sierra::nalu::AlgTraitsHex8> elemAlg(
    helperObjs.realm, partVec_[0], tke_, dkdx_);
  elemAlg.execute();

  {
    std::vector<double> expectedValues = {
      4,  4,  4, -4,  6,  6,
      6, -4,  6, -6, -6,  8,
      6,  6, -4, -6,  8, -6,
      8, -6, -6, -8, -8, -8,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, ngp_nodal_grad_elem_shifted)
{
  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradElem
  // sierra::nalu::AssembleNodalGradElemAlgorithm elemAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_, useShifted);
  // elemAlg.execute();
  sierra::nalu::ScalarNodalGradElemAlg<sierra::nalu::AlgTraitsHex8> elemAlg(
    helperObjs.realm, partVec_[0], tke_, dkdx_, useShifted);
  elemAlg.execute();

  {
    std::vector<double> expectedValues = {
      2, 2, 2, -2, 6, 6,
      6, -2, 6, -6, -6, 10,
      6, 6, -2, -6, 10, -6,
      10, -6, -6, -10, -10, -10
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, ngp_nodal_grad_bndry)
{
  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = false;

  auto* part = meta_.get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradBoundary
  // sierra::nalu::AssembleNodalGradBoundaryAlgorithm elemAlg(
  //   helperObjs.realm, part, tke_, dkdx_, useShifted);
  // elemAlg.execute();
  sierra::nalu::ScalarNodalGradBndryElemAlg<sierra::nalu::AlgTraitsQuad4> elemAlg(
    helperObjs.realm, part, tke_, dkdx_, useShifted);
  elemAlg.execute();

  {
    std::vector<double> expectedValues = {-2, -4, -4, -6};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], 0.0, tol);
        EXPECT_NEAR(dkdx[1], 0.0, tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, ngp_nodal_grad_bndry_shifted)
{
  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = true;

  auto* part = meta_.get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradBounndary
  // sierra::nalu::AssembleNodalGradBoundaryAlgorithm elemAlg(
  //   helperObjs.realm, part, tke_, dkdx_, useShifted);
  // elemAlg.execute();
  sierra::nalu::ScalarNodalGradBndryElemAlg<sierra::nalu::AlgTraitsQuad4> elemAlg(
    helperObjs.realm, part, tke_, dkdx_, useShifted);
  elemAlg.execute();

  {
    std::vector<double> expectedValues = {-0, -4, -4, -8};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], 0.0, tol);
        EXPECT_NEAR(dkdx[1], 0.0, tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}
