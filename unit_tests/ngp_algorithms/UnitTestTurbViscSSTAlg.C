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

#include "ngp_algorithms/TurbViscSSTAlg.h"
#include "ngp_algorithms/TurbViscSSTLRAlg.h"

#include <cmath>

namespace {

class APGTurbViscSSTTest
  : public SSTKernelHex8Mesh,
    public ::testing::WithParamInterface<bool>
{
public:
  APGTurbViscSSTTest()
    : sensor_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "lamda0L")),
      apgVelocity_(&meta_->declare_field<double>(
        stk::topology::NODE_RANK, "velocity"))
  {
    stk::mesh::put_field_on_mesh(*sensor_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *apgVelocity_, meta_->universal_part(), spatialDim_, nullptr);
  }

  sierra::kynema_ugf::ScalarFieldType* sensor_;
  sierra::kynema_ugf::VectorFieldType* apgVelocity_;

  struct Case
  {
    double dvnn, gamma, sdr, distance, sensor, plainVisc, transitionVisc;
  };

  void run_cases(
    sierra::kynema_ugf::Realm& realm, const std::vector<Case>& cases)
  {
    sierra::kynema_ugf::TurbViscSSTAlg alg(realm, partVec_[0], tvisc_);
    const auto& fieldMgr = realm.ngp_field_manager();
    auto& ngpTvisc =
      fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
    auto& ngpSensor =
      fieldMgr.get_field<double>(sensor_->mesh_meta_data_ordinal());
    const auto& buckets =
      bulk_->get_buckets(stk::topology::NODE_RANK, meta_->universal_part());
    for (const auto& c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "dvnn=" << c.dvnn << ", gamma=" << c.gamma
                   << ", sdr=" << c.sdr << ", distance=" << c.distance);
      stk::mesh::field_fill(1.0, *density_);
      stk::mesh::field_fill(1.0, *visc_);
      stk::mesh::field_fill(1.0, *tke_);
      stk::mesh::field_fill(c.sdr, *sdr_);
      stk::mesh::field_fill(c.distance, *minDistance_);
      stk::mesh::field_fill(c.gamma, *gamma_intermittency_);
      const double normal[3]{0.0, 1.0, 0.0};
      const double gradient[3]{0.0, c.dvnn, 0.0};
      // The large normal velocity must not enter the wall-parallel Reynolds
      // number.
      const double velocity[3]{100.0, 10000.0, 0.0};
      const double strain[9]{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      stk::mesh::field_fill_component(normal, *dwalldistdx_);
      stk::mesh::field_fill_component(gradient, *dnDotVdx_);
      stk::mesh::field_fill_component(velocity, *apgVelocity_);
      stk::mesh::field_fill_component(strain, *dudx_);
      for (auto* field :
           {density_, visc_, tke_, sdr_, minDistance_, gamma_intermittency_,
            dwalldistdx_, dnDotVdx_, apgVelocity_, dudx_}) {
        field->modify_on_host();
        field->sync_to_device();
      }

      alg.execute();
      ngpTvisc.sync_to_host();
      ngpSensor.sync_to_host();
      for (const auto* bucket : buckets)
        for (const auto node : *bucket) {
          EXPECT_NEAR(
            stk::mesh::field_data(*sensor_, node)[0], c.sensor, 1.0e-14);
          EXPECT_NEAR(
            stk::mesh::field_data(*tvisc_, node)[0],
            GetParam() ? c.transitionVisc : c.plainVisc, 1.0e-14);
        }
    }
  }
};

TEST_P(APGTurbViscSSTTest, sensor_and_viscosity)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  auto& options = *helperObjs.realm.solutionOptions_;
  options.apgCorrection_ = true;
  options.gammaEqActive_ = GetParam();
  EXPECT_DOUBLE_EQ(
    options.turbModelConstantMap_[sierra::kynema_ugf::TM_aOneApg], 0.275);
  EXPECT_DOUBLE_EQ(
    options.turbModelConstantMap_[sierra::kynema_ugf::TM_sThresApg], -5.0e-4);

  const double baseline = 0.31 / std::sqrt(8.0);
  const double corrected = 0.275 / std::sqrt(8.0);
  run_cases(
    helperObjs.realm,
    {
      {20.0, 1.0, 1.0, 1.0, -0.1386, corrected, corrected},
      {0.0, 1.0, 1.0, 1.0, 0.0128, baseline, baseline},
      {20.0, 1.0, 100.0, 1.0, -0.1386, 0.01, 0.01},
      {6.0, 0.0, 1.0, 1.0, -0.03262, corrected, baseline},
      {6.0, 1.0, 1.0, 1.0, -0.03262, corrected, corrected},
      {6.0, 0.5, 1.0, 1.0, -0.03262, corrected, baseline},
      {1000.0, 1.0, 1.0, 1.0, -1.0, corrected, corrected},
      {-1000.0, 1.0, 1.0, 1.0, 1.0, baseline, baseline},
      {20.0, 1.0, 0.0, 0.0, 0.0128, baseline, baseline},
    });
}

TEST_P(APGTurbViscSSTTest, user_constants)
{
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  auto& options = *helperObjs.realm.solutionOptions_;
  options.apgCorrection_ = true;
  options.gammaEqActive_ = GetParam();
  options.turbModelConstantMap_[sierra::kynema_ugf::TM_aOneApg] = 0.2;
  options.turbModelConstantMap_[sierra::kynema_ugf::TM_sThresApg] = -2.0e-3;

  // With sThres_apg = -2e-3 and Re = 100, the turbulent threshold becomes
  // -0.0632, so lambda = -0.03262 no longer triggers the correction.
  const double baseline = 0.31 / std::sqrt(8.0);
  const double corrected = 0.2 / std::sqrt(8.0);
  run_cases(
    helperObjs.realm, {
                        {20.0, 1.0, 1.0, 1.0, -0.1386, corrected, corrected},
                        {6.0, 1.0, 1.0, 1.0, -0.03262, baseline, baseline},
                      });
}

INSTANTIATE_TEST_SUITE_P(
  SSTAndSSTGamma, APGTurbViscSSTTest, ::testing::Bool());

} // namespace

TEST_F(SSTKernelHex8Mesh, NGP_turb_visc_sst_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  SSTKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::kynema_ugf::TurbViscSSTAlg TurbViscSSTAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscSSTAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      1,
      0.58778525229247314,
      0.82554938136626144,
      0.33398788772269555,
      0.58778525229247314,
      0.34549150281252627,
      0.32219673776462515,
      0.19411612710296283};

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_turb_visc_sstlr_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  SSTKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::kynema_ugf::TurbViscSSTLRAlg TurbViscSSTLRAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscSSTLRAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      1,
      0.58778525229247314,
      0.82554938136626144,
      0.33398788772269555,
      0.58778525229247314,
      0.17412848496705169,
      0.32219673776462515,
      0.13137855541068333};

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}

TEST_F(AMSKernelHex8Mesh, NGP_turb_visc_sstams_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  AMSKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::kynema_ugf::TurbViscSSTAlg TurbViscSSTAlg(
    helperObjs.realm, partVec_[0], tvisc_, true);

  TurbViscSSTAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      1, 1, 1.4045084971874737,  0.62203263379915574,
      1, 1, 0.93257499863740057, 0.57277822202446083,
    };

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}
