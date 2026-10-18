// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/SANuTildaNodeKernel.h"
#include "node_kernels/SANuTildaAFTNodeKernel.h"

namespace {

// Spatially uniform fields, so every node has the same gold value:
// rho = 1, mu = 0.01, d = 0.1, dual volume = 0.125, du/dy = 2 (Omega = 2),
// dnutdx = (0.1, 0.2, 0). Golds computed independently in Python.
namespace sa_golds {
// chi = 0.5: standard STilda branch
namespace sa_std {
static constexpr double rhs = 0.0053577720395066022;
static constexpr double lhs = 0.44466214581461916;
} // namespace sa_std
// chi = 5: modified STilda branch (Sbar < -Cv2 * Omega)
namespace sa_mod {
static constexpr double rhs = -0.19696081789349024;
static constexpr double lhs = 8.1151637158240035;
} // namespace sa_mod
// nuTilda = -0.02: SA-neg branch
namespace sa_neg {
static constexpr double rhs = 0.022162089083878643;
static constexpr double lhs = 1.6263089083878639;
} // namespace sa_neg
// AFT with gammaTilda = -0.5
namespace aft_std {
static constexpr double rhs = 0.0053832692827119194;
static constexpr double lhs = 0.47141203577617979;
} // namespace aft_std
namespace aft_mod {
static constexpr double rhs = -0.18514916826959443;
static constexpr double lhs = 7.6410246136146212;
} // namespace aft_mod
namespace aft_neg {
static constexpr double rhs = 0.022162089083878643;
static constexpr double lhs = 1.6263089083878639;
} // namespace aft_neg
} // namespace sa_golds

class SAKernelHex8Mesh : public LowMachKernelHex8Mesh
{
public:
  SAKernelHex8Mesh()
    : LowMachKernelHex8Mesh(),
      nuTilda_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "sa_nu_tilda")),
      gammaTilda_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "aft_gamma_tilda")),
      visc_(
        &meta_->declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
      minDistance_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "minimum_distance_to_wall")),
      dudx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx")),
      dnutdx_(&meta_->declare_field<double>(stk::topology::NODE_RANK, "dnutdx"))
  {
    stk::mesh::put_field_on_mesh(*nuTilda_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *gammaTilda_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*visc_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *minDistance_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *dudx_, meta_->universal_part(), spatialDim_ * spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *dnutdx_, meta_->universal_part(), spatialDim_, nullptr);
  }

  void fill_sa_fields(const double nuTilda, const double gammaTilda = 0.0)
  {
    LowMachKernelHex8Mesh::fill_mesh_and_init_fields();

    const double dudx[9] = {0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double dnutdx[3] = {0.1, 0.2, 0.0};

    stk::mesh::field_fill(1.0, *density_);
    stk::mesh::field_fill(nuTilda, *nuTilda_);
    stk::mesh::field_fill(gammaTilda, *gammaTilda_);
    stk::mesh::field_fill(0.01, *visc_);
    stk::mesh::field_fill(0.1, *minDistance_);
    stk::mesh::field_fill_component(dudx, *dudx_);
    stk::mesh::field_fill_component(dnutdx, *dnutdx_);

    for (stk::mesh::FieldBase* fld : std::vector<stk::mesh::FieldBase*>{
           density_, nuTilda_, gammaTilda_, visc_, minDistance_, dudx_,
           dnutdx_}) {
      fld->modify_on_host();
      fld->sync_to_device();
    }
  }

  template <typename KernelType>
  void run_and_check(const double rhsGold, const double lhsGold)
  {
    solnOpts_.meshMotion_ = false;
    solnOpts_.externalMeshDeformation_ = false;
    solnOpts_.initialize_turbulence_constants();

    unit_test_utils::NodeHelperObjects helperObjs(
      bulk_, stk::topology::HEX_8, 1, partVec_[0]);

    helperObjs.nodeAlg->add_kernel<KernelType>(*meta_);
    helperObjs.execute();

    EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 8u);
    EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 8u);
    EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 8u);
    EXPECT_EQ(helperObjs.linsys->hostNumSumIntoCalls_(0), 8u);

    double lhs[8][8] = {};
    for (int i = 0; i < 8; ++i)
      lhs[i][i] = lhsGold;

    unit_test_kernel_utils::expect_all_near(
      helperObjs.linsys->rhs_, rhsGold, 1.0e-12);
    unit_test_kernel_utils::expect_all_near<8>(
      helperObjs.linsys->lhs_, lhs, 1.0e-12);
  }

  sierra::kynema_ugf::ScalarFieldType* nuTilda_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* gammaTilda_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* visc_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* minDistance_{nullptr};
  sierra::kynema_ugf::TensorFieldType* dudx_{nullptr};
  sierra::kynema_ugf::VectorFieldType* dnutdx_{nullptr};
};

} // namespace

TEST_F(SAKernelHex8Mesh, NGP_sa_nutilda_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(0.005);
  run_and_check<sierra::kynema_ugf::SANuTildaNodeKernel>(
    sa_golds::sa_std::rhs, sa_golds::sa_std::lhs);
}

TEST_F(SAKernelHex8Mesh, NGP_sa_nutilda_modified_stilda_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(0.05);
  run_and_check<sierra::kynema_ugf::SANuTildaNodeKernel>(
    sa_golds::sa_mod::rhs, sa_golds::sa_mod::lhs);
}

TEST_F(SAKernelHex8Mesh, NGP_sa_nutilda_negative_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(-0.02);
  run_and_check<sierra::kynema_ugf::SANuTildaNodeKernel>(
    sa_golds::sa_neg::rhs, sa_golds::sa_neg::lhs);
}

TEST_F(SAKernelHex8Mesh, NGP_sa_aft_nutilda_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(0.005, -0.5);
  run_and_check<sierra::kynema_ugf::SANuTildaAFTNodeKernel>(
    sa_golds::aft_std::rhs, sa_golds::aft_std::lhs);
}

TEST_F(SAKernelHex8Mesh, NGP_sa_aft_nutilda_modified_stilda_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(0.05, -0.5);
  run_and_check<sierra::kynema_ugf::SANuTildaAFTNodeKernel>(
    sa_golds::aft_mod::rhs, sa_golds::aft_mod::lhs);
}

TEST_F(SAKernelHex8Mesh, NGP_sa_aft_nutilda_negative_node)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_sa_fields(-0.02, -0.5);
  run_and_check<sierra::kynema_ugf::SANuTildaAFTNodeKernel>(
    sa_golds::aft_neg::rhs, sa_golds::aft_neg::lhs);
}
