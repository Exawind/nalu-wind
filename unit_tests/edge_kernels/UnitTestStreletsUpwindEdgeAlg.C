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
          velocity: 1.0
          turbulent_ke: 1.0
          specific_dissipation_rate: 1.0

      - alpha_upw:
          velocity: 1.0
          turbulent_ke: 1.0
          specific_dissipation_rate: 1.0

      - upw_factor:
          velocity: 0.0
          turbulent_ke: 0.0
          specific_dissipation_rate: 0.0

     )inp";

  fill_mesh_and_init_fields();

  YAML::Node realm_node = YAML::Load(realmInput);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0], true,
    unit_test_utils::get_default_inputs(), realm_node[0]);

  helperObjs.realm.interiorPartVec_.push_back(partVec_[0]);
  ASSERT_EQ(0.0, helperObjs.realm.solutionOptions_->get_upw_factor("velocity"));

  // init fields

  sierra::nalu::StreletsUpwindEdgeAlg streletsUpw(
    helperObjs.realm, partVec_[0]);
  ASSERT_NO_THROW(streletsUpw.execute());
}

} // namespace nalu
} // namespace sierra