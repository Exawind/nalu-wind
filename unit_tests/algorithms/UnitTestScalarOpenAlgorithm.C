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

  const int nprocs = this->bulk().parallel_size();
  std::string proc_num = std::to_string(nprocs);
  std::string meshSpec = "generated:1x1x"+proc_num+"|sideset:xXyYzZ";
  fill_mesh_and_init_fields(meshSpec);

  stk::mesh::Part* sidePart = meta().get_part("surface_2");

  unit_test_utils::FaceElemHelperObjects helperObjs(*realm.bulkData_, stk::topology::QUAD_4, stk::topology::HEX_8, 1, sidePart);

  sierra::nalu::AssembleScalarEdgeOpenSolverAlgorithm alg(realm, sidePart, &helperObjs.eqSystem, tke_, tkebc_, dkdx_, tvisc_);
  alg.execute();
}

