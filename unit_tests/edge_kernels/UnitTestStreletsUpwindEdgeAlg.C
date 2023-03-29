// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
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

#include "edge_kernels/StreletsUpwindEdgeAlg.h"
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

TEST_F(SSTKernelHex8Mesh, StreletsUpwindComputation)
{
  const char* realmInput = R"inp(- name: unitTestRealm
  use_edges: yes
 
  equation_systems:
    name: theEqSys
    max_iterations: 2
    
    solver_system_specification:
      velocity: solve_scalar
      turbulent_ke: solve_scalar
      specific_dissipation_rate: solve_scalar
      pressure: solve_cont
      ndtw: solve_cont

    systems:
      - WallDistance:
          name: myNDTW
          max_iterations: 1
          convergence_tolerance: 1.0e-8

      - LowMachEOM:
          name: myLowMach
          max_iterations: 1
          convergence_tolerance: 1.0e-8

      - ShearStressTransport:
          name: mySST
          max_iterations: 1
          convergence_tolerance: 1.0e-8

  time_step_control:
    target_courant: 2.0
    time_step_change_factor: 1.2 

  solution_options:
    name: myOptions
    turbulence_model: sst_iddes
    projected_timescale_type: momentum_diag_inv

    options:
      - hybrid_factor:
          turbulent_ke: 1.0
          specific_dissipation_rate: 1.0

      - alpha_upw:
          velocity: 1.0
          turbulent_ke: 1.0
          specific_dissipation_rate: 1.0

      - upw_factor:
          velocity: 1.0
          turbulent_ke: 0.0
          specific_dissipation_rate: 0.0

     )inp";

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmInput);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0], true,
    unit_test_utils::get_default_inputs(), realm_node[0]);

  Realm& realm = helperObjs.realm;
  realm.interiorPartVec_.push_back(partVec_[0]);

  // init fields
  const double tanhOne = std::tanh(1.0);

  // set all turb constants to 1 for the Strelets comp
  // make |S|=4 and |\Omega| = 2
  const double D = std::sqrt(8.0 / 3.0);
  const double L = std::sqrt(1.0 / 3.0);
  const double U = -1.0 * L;
  const double dudx[9] = {D, U, U, L, D, U, L, L, D};

  stk::mesh::field_fill_component(dudx, *dudx_);
  dudx_->modify_on_host();
  dudx_->sync_to_device();

  // we need to make a consistent velocity field to pair with
  const stk::mesh::Selector sel = stk::mesh::selectField(*velocity_);

  for (const auto* ib : bulk_->get_buckets(stk::topology::NODE_RANK, sel)) {
    const auto& b = *ib;
    const size_t length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity node = b[k];
      const double* x = stk::mesh::field_data(*coordinates_, node);
      double* vel = stk::mesh::field_data(*velocity_, node);
      vel[0] = D * x[0] + U * x[1] + U * x[2];
      vel[1] = L * x[0] + D * x[1] + U * x[2];
      vel[2] = L * x[0] + L * x[1] + D * x[2];
    }
  }
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  // now we can simplify the equations to give us a nice-ish number for the
  // upwind computation
  // reference to M. Strelets, "Detached Eddy simulations of massively seperated
  // flows", AIAA, 2001
  // eq 12 of the paper B = CH3*8/10 with these S and \Omega
  // values so let's set
  realm.solutionOptions_->turbModelConstantMap_[TM_ch3] = 10.0 / 8.0;
  // K = \sqrt(10)
  // eq 11 l_turb = (\nu_t + \nu)/sqrt(C_\mu^3/2*K) which we can make unity as
  // well by setting the denominator to 10
  realm.solutionOptions_->turbModelConstantMap_[TM_cMu] = 10.0;
  // and setting the numerator to 10
  stk::mesh::field_fill(5.0, *visc_);
  stk::mesh::field_fill(5.0, *tvisc_);
  stk::mesh::field_fill(1.0, *density_);
  visc_->modify_on_host();
  visc_->sync_to_device();
  tvisc_->modify_on_host();
  tvisc_->sync_to_device();
  density_->modify_on_host();
  density_->sync_to_device();
  // now eq 10 becomes
  // A = CH2 * (CDES * \Delta / tanh(1.0) - 0.5) which we can also make unity
  // (1.0/CH2 +0.5) * tanh(1.0) = CDES *\Delta by setting CH2 to 2 \Delta to
  // tanh(1.0) and CDES to 1.0
  realm.solutionOptions_->turbModelConstantMap_[TM_ch2] = 2.0;
  // CDES comes from equation 8 of the paper
  // CDES = (1-F1) * CDES_keps + F1 * CDES_komeg
  const double F1 = 0.25;
  stk::mesh::field_fill(F1, *fOneBlend_);
  fOneBlend_->modify_on_host();
  fOneBlend_->sync_to_device();

  realm.solutionOptions_->turbModelConstantMap_[TM_cDESke] = 0.4 / (1.0 - F1);
  realm.solutionOptions_->turbModelConstantMap_[TM_cDESkw] = 0.6 / F1;
  const double cDESke =
    realm.solutionOptions_->turbModelConstantMap_[TM_cDESke];
  const double cDESkw =
    realm.solutionOptions_->turbModelConstantMap_[TM_cDESkw];
  ASSERT_DOUBLE_EQ(1.0, (1.0 - F1) * cDESke + F1 * cDESkw);

  stk::mesh::field_fill(tanhOne, *maxLengthScale_);
  maxLengthScale_->modify_on_host();
  maxLengthScale_->sync_to_device();

  // so finally our expected value is tanh(1.0) from eq 9

  sierra::nalu::StreletsUpwindEdgeAlg streletsUpw(
    helperObjs.realm, partVec_[0]);
  ASSERT_NO_THROW(streletsUpw.execute());

  pecletFactor_->sync_to_host();

  // check on host for values
  const auto rank = NaluEnv::self().parallel_rank();
  for (const auto* ib : bulk_->get_buckets(
         stk::topology::EDGE_RANK, meta_->locally_owned_part())) {
    const auto& b = *ib;
    const size_t length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity edge = b[k];
      const auto* nodes = bulk_->begin_nodes(edge);
      const auto gEdge = bulk_->identifier(edge);
      const double* x1 = stk::mesh::field_data(*coordinates_, nodes[0]);
      const double* x2 = stk::mesh::field_data(*coordinates_, nodes[1]);
      const auto nId1 = bulk_->identifier(nodes[0]);
      const auto nId2 = bulk_->identifier(nodes[1]);
      const double fieldVal = *stk::mesh::field_data(*pecletFactor_, edge);

      EXPECT_NEAR(tanhOne, fieldVal, 1e-12)
        << "EdgeID " << gEdge << ", "
        << " Rank: " << rank << std::endl
        << "Node " << nId1 << ", " << x1[0] << ", " << x1[1] << ", " << x1[2]
        << std::endl
        << "Node " << nId2 << ", " << x2[0] << ", " << x2[1] << ", " << x2[2];
    }
  }
}

} // namespace nalu
} // namespace sierra
