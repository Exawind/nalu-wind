// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"
#include "ngp_algorithms/UnitTestNgpAlgUtils.h"

#include "AlgTraits.h"
#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_algorithms/NodalGradElemAlg.h"
#include "ngp_algorithms/NodalGradBndryElemAlg.h"
#include "ngp_algorithms/NodalGradAlgDriver.h"

#include "stk_mesh/base/CreateEdges.hpp"

TEST_F(SSTKernelHex8Mesh, NGP_nodal_grad_edge)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *tke_, xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradEdge
  // sierra::nalu::AssembleNodalGradEdgeAlgorithm edgeAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_);
  // edgeAlg.execute();

  sierra::nalu::ScalarNodalGradAlgDriver algDriver(
    helperObjs.realm, tke_->name(), "dkdx");
  algDriver.register_edge_algorithm<sierra::nalu::ScalarNodalGradEdgeAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", tke_, dkdx_);
  algDriver.execute();

  {
    std::vector<double> expectedValues = {2,  2,  2,  -2, 6,  6,   6,   -2,
                                          6,  -6, -6, 10, 6,  6,   -2,  -6,
                                          10, -6, 10, -6, -6, -10, -10, -10};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(MomentumKernelHex8Mesh, NGP_nodal_grad_edge_vec)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *velocity_, xCoeff, yCoeff, zCoeff);
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *dudx_);
  dudx_->modify_on_host();
  dudx_->sync_to_device();

  // Reference values from original AssembleNodalGradEdge
  // sierra::nalu::AssembleNodalGradUEdgeAlgorithm edgeAlg(
  //   helperObjs.realm, partVec_[0], velocity_, dudx_);
  // edgeAlg.execute();

  sierra::nalu::TensorNodalGradAlgDriver algDriver(
    helperObjs.realm, velocity_->name(), "dudx");
  algDriver.register_edge_algorithm<sierra::nalu::TensorNodalGradEdgeAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", velocity_, dudx_);
  algDriver.execute();

  {
    // Test the `du_i/dx_i \delta_ii` values
    std::vector<double> expectedValues = {
      2, 2, 2,  -2, 2, 2,  2, -2, 2,  -2, -2, 2,
      2, 2, -2, -2, 2, -2, 2, -2, -2, -2, -2, -2,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dudx = stk::mesh::field_data(*dudx_, node);
        for (int i1 = 0; i1 < 3; ++i1)
          EXPECT_NEAR(dudx[i1 * 3 + i1], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_nodal_grad_elem)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *tke_, xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradElem
  // sierra::nalu::AssembleNodalGradElemAlgorithm elemAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_);
  // elemAlg.execute();

  sierra::nalu::ScalarNodalGradAlgDriver algDriver(
    helperObjs.realm, tke_->name(), "dkdx");
  algDriver.register_elem_algorithm<sierra::nalu::ScalarNodalGradElemAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", tke_, dkdx_);
  algDriver.execute();

  {
    std::vector<double> expectedValues = {
      4, 4, 4,  -4, 6, 6,  6, -4, 6,  -6, -6, 8,
      6, 6, -4, -6, 8, -6, 8, -6, -6, -8, -8, -8,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_nodal_grad_elem_shifted)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *tke_, xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradElem
  // sierra::nalu::AssembleNodalGradElemAlgorithm elemAlg(
  //   helperObjs.realm, partVec_[0], tke_, dkdx_, useShifted);
  // elemAlg.execute();

  sierra::nalu::ScalarNodalGradAlgDriver algDriver(
    helperObjs.realm, tke_->name(), "dkdx");
  algDriver.register_elem_algorithm<sierra::nalu::ScalarNodalGradElemAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", tke_, dkdx_, useShifted);
  algDriver.execute();

  {
    std::vector<double> expectedValues = {2,  2,  2,  -2, 6,  6,   6,   -2,
                                          6,  -6, -6, 10, 6,  6,   -2,  -6,
                                          10, -6, 10, -6, -6, -10, -10, -10};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[1], expectedValues[ii++], tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(MomentumKernelHex8Mesh, NGP_nodal_grad_elem_vec)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = false;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *velocity_, xCoeff, yCoeff, zCoeff);
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *dudx_);
  dudx_->modify_on_host();
  dudx_->sync_to_device();

  // Reference values from original AssembleNodalGradEdge
  // sierra::nalu::AssembleNodalGradUElemAlgorithm elemAlg(
  //   helperObjs.realm, partVec_[0], velocity_, dudx_, useShifted);
  // elemAlg.execute();

  sierra::nalu::TensorNodalGradAlgDriver algDriver(
    helperObjs.realm, velocity_->name(), "dudx");
  algDriver.register_elem_algorithm<sierra::nalu::TensorNodalGradElemAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", velocity_, dudx_,
    useShifted);
  algDriver.execute();

  {
    // Test the `du_i/dx_i \delta_ii` values
    std::vector<double> expectedValues = {
      2, 2, 2,  -2, 2, 2,  2, -2, 2,  -2, -2, 2,
      2, 2, -2, -2, 2, -2, 2, -2, -2, -2, -2, -2,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dudx = stk::mesh::field_data(*dudx_, node);
        for (int i1 = 0; i1 < 3; ++i1)
          EXPECT_NEAR(dudx[i1 * 3 + i1], expectedValues[ii++], tol);
      }
  }
}

TEST_F(MomentumKernelHex8Mesh, NGP_nodal_grad_elem_shifted_vec)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *velocity_, xCoeff, yCoeff, zCoeff);
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *dudx_);
  dudx_->modify_on_host();
  dudx_->sync_to_device();

  // Reference values from original AssembleNodalGradEdge
  // sierra::nalu::AssembleNodalGradUEdgeAlgorithm edgeAlg(
  //   helperObjs.realm, partVec_[0], velocity_, dudx_);
  // edgeAlg.execute();

  sierra::nalu::TensorNodalGradAlgDriver algDriver(
    helperObjs.realm, velocity_->name(), "dudx");
  algDriver.register_elem_algorithm<sierra::nalu::TensorNodalGradElemAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "nodal_grad", velocity_, dudx_,
    useShifted);
  algDriver.execute();

  {
    // Test the `du_i/dx_i \delta_ii` values
    std::vector<double> expectedValues = {
      2, 2, 2,  -2, 2, 2,  2, -2, 2,  -2, -2, 2,
      2, 2, -2, -2, 2, -2, 2, -2, -2, -2, -2, -2,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dudx = stk::mesh::field_data(*dudx_, node);
        for (int i1 = 0; i1 < 3; ++i1)
          EXPECT_NEAR(dudx[i1 * 3 + i1], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_nodal_grad_bndry)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = false;

  auto* part = meta_->get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *tke_, xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradBoundary
  // sierra::nalu::AssembleNodalGradBoundaryAlgorithm elemAlg(
  //   helperObjs.realm, part, tke_, dkdx_, useShifted);
  // elemAlg.execute();

  auto* surfPart = part->subsets()[0];
  sierra::nalu::ScalarNodalGradAlgDriver algDriver(
    helperObjs.realm, tke_->name(), "dkdx");
  algDriver.register_face_algorithm<sierra::nalu::ScalarNodalGradBndryElemAlg>(
    sierra::nalu::WALL, surfPart, "nodal_grad", tke_, dkdx_, useShifted);
  algDriver.execute();

  {
    std::vector<double> expectedValues = {-2, -4, -4, -6};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], 0.0, tol);
        EXPECT_NEAR(dkdx[1], 0.0, tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_nodal_grad_bndry_shifted)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = true;

  auto* part = meta_->get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *tke_, xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *dkdx_);

  // Reference values from original AssembleNodalGradBounndary
  // sierra::nalu::AssembleNodalGradBoundaryAlgorithm elemAlg(
  //   helperObjs.realm, part, tke_, dkdx_, useShifted);
  // elemAlg.execute();

  auto* surfPart = part->subsets()[0];
  sierra::nalu::ScalarNodalGradAlgDriver algDriver(
    helperObjs.realm, tke_->name(), "dkdx");
  algDriver.register_face_algorithm<sierra::nalu::ScalarNodalGradBndryElemAlg>(
    sierra::nalu::WALL, surfPart, "nodal_grad", tke_, dkdx_, useShifted);
  algDriver.execute();

  {
    std::vector<double> expectedValues = {-0, -4, -4, -8};

    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dkdx = stk::mesh::field_data(*dkdx_, node);
        EXPECT_NEAR(dkdx[0], 0.0, tol);
        EXPECT_NEAR(dkdx[1], 0.0, tol);
        EXPECT_NEAR(dkdx[2], expectedValues[ii++], tol);
      }
  }
}

TEST_F(MomentumKernelHex8Mesh, NGP_nodal_grad_bndry_elem_vec)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;
  const bool useShifted = false;

  auto* part = meta_->get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(
    *bulk_, *coordinates_, *velocity_, xCoeff, yCoeff, zCoeff);
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *dudx_);
  dudx_->modify_on_host();
  dudx_->sync_to_device();

  // Reference values from original
  // sierra::nalu::AssembleNodalGradUBoundaryAlgorithm elemAlg(
  //   helperObjs.realm, part, velocity_, dudx_, useShifted);
  // elemAlg.execute();

  auto* surfPart = part->subsets()[0];
  sierra::nalu::TensorNodalGradAlgDriver algDriver(
    helperObjs.realm, velocity_->name(), "dudx");
  algDriver.register_face_algorithm<sierra::nalu::TensorNodalGradBndryElemAlg>(
    sierra::nalu::WALL, surfPart, "nodal_grad", velocity_, dudx_, useShifted);
  algDriver.execute();

  {
    // du_i/dz components
    std::vector<double> expectedValues = {
      -1, -1, 0, -1, -3, 0, -3, -1, 0, -3, -3, 0,
    };

    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* dudx = stk::mesh::field_data(*dudx_, node);
        for (int i1 = 0; i1 < 3; ++i1)
          EXPECT_NEAR(dudx[i1 * 3 + 2], expectedValues[ii++], tol);
      }
  }
}
