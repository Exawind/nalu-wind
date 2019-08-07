/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "UnitTestAlgorithm.h"
#include "UnitTestKokkosUtils.h"
#include "UnitTestFieldUtils.h"
#include "UnitTestAlgorithmUtils.h"
#include "UnitTestHelperObjects.h"

#include "AssembleScalarEdgeOpenSolverAlgorithm.h"

TEST_F(TestTurbulenceAlgorithm, scalarOpenEdgeAlgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields("generated:1x1x1|sideset:xXyYzZ");

  stk::mesh::Part* sidePart = meta().get_part("surface_2");

  unit_test_utils::FaceElemHelperObjects helperObjs(*realm.bulkData_, stk::topology::QUAD_4, stk::topology::HEX_8, 1, sidePart);

  // Execute
  sierra::nalu::AssembleScalarEdgeOpenSolverAlgorithm alg(realm, sidePart, &helperObjs.eqSystem, tke_, tkebc_, dkdx_, tvisc_);
  alg.execute();

  helperObjs.print_lhs_and_rhs();
}

