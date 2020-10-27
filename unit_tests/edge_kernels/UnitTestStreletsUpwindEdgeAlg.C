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

namespace sierra {
namespace nalu {

TEST_F(SSTKernelHex8Mesh, StreletsUpwindComputation)
{
  fill_mesh_and_init_fields();
  solnOpts_.turbulenceModel_ = SST_IDDES;
  solnOpts_.upwMap_["velocity"] = 0.0;
  ASSERT_EQ(0.0, solnOpts_.get_upw_factor("velocity"));

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0], true);

  helperObjs.realm.interiorPartVec_.push_back(partVec_[0]);
  ASSERT_EQ(0.0, helperObjs.realm.solutionOptions_->get_upw_factor("velocity"));

  // init fields

  sierra::nalu::StreletsUpwindEdgeAlg streletsUpw(
    helperObjs.realm, partVec_[0]);
  ASSERT_NO_THROW(streletsUpw.execute());
}

} // namespace nalu
} // namespace sierra