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

#include "kernel/ScalarFluxBCElemKernel.h"

TEST_F(EnthalpyABLKernelHex8Mesh, NGP_heat_flux_bc)
{
  if (bulk_.parallel_size() > 1) return;

  const std::string meshSpec;
  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(meshSpec, doPerturb, generateSidesets);

  auto* part = meta_.get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);

  const std::string coordsName = "coordinates";
  const bool useShifted = false;

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::ScalarFluxBCElemKernel<sierra::nalu::AlgTraitsQuad4>(
      bulk_, heatFluxBC_, coordsName, useShifted,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Register the kernel for execution
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  // Populate LHS and RHS
  helperObjs.execute();


  // heatflux set to 100.0 and area_mag is 0.25
  const double rhsExact = 25.0;
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 4u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 4u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 4u);

  unit_test_kernel_utils::expect_all_near(
    helperObjs.linsys->rhs_, rhsExact, 1.0e-12);
  unit_test_kernel_utils::expect_all_near<4>(
    helperObjs.linsys->lhs_, 0.0, 1.0e-12);
}
