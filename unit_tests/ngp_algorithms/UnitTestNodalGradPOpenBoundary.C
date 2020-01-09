// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "AlgTraits.h"

#include "stk_mesh/base/CreateEdges.hpp"
#include "kernels/UnitTestKernelUtils.h"

#include "UnitTestHelperObjects.h"

#include "ngp_algorithms/UnitTestNgpAlgUtils.h"
#include "ngp_algorithms/NodalGradPOpenBoundaryAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/GeometryInteriorAlg.h"

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/GeometryAlgDriver.h"

#include <stk_mesh/base/SkinBoundary.hpp>

TEST_F(LowMachKernelHex8Mesh, NGP_nodal_grad_popen)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  const std::string meshSpec = "generated:20x20x20";
  const bool doPerturb = false;
  const bool generateSidesets = true;

  fill_mesh_and_init_fields(meshSpec, doPerturb, generateSidesets);

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  const double zero= 0.0;
  const double one = 1.0;
  const double two = 2.0;
  const double oneVecTwelve[12] = {one, one, one, one, one, one, one, one, one, one, one, one};
  stk::mesh::field_fill(zero, *dpdx_);
  stk::mesh::field_fill(two, *dnvField_);
  stk::mesh::field_fill(one, *pressure_);
  stk::mesh::field_fill_component(oneVecTwelve, *exposedAreaVec_);

  stk::mesh::Part* surface1 = meta_.get_part("surface_1");
  {
    sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
    geomAlgDriver.register_face_algorithm<sierra::nalu::GeometryBoundaryAlg>(
      sierra::nalu::WALL, surface1, "geometry");
    geomAlgDriver.execute();
  }

  stk::mesh::field_fill(2.0, *dnvField_);

  {
    helperObjs.realm.solutionOptions_->activateOpenMdotCorrection_ = true;
    sierra::nalu::ScalarNodalGradAlgDriver algDriver(helperObjs.realm, "dpdx");
    algDriver.register_face_elem_algorithm<sierra::nalu::NodalGradPOpenBoundary>(
      sierra::nalu::OPEN, surface1, stk::topology::HEX_8, "nodal_grad_pressure_open_boundary", false);
    algDriver.execute();
  }

  {
    const double dpdxref[6][3] = {{-0.25, 0.0,  -0.25},
                                  {-0.25, 0.0,   0.25},
                                  {-0.25,-0.25,  0.0},
                                  {-0.25, 0.25,  0.0},
                                  {-0.5,  0.0,   0.0},
                                  { 0.0,  0.0,   0.0}};

    const double tol = 1.0e-16;

    ngp::Field<double> ngpdpdx(bulk_, *dpdx_);
    ngp::Field<double> ngpcord(bulk_, *coordinates_);
    ngpdpdx.modify_on_device();
    ngpdpdx.sync_to_host();
    ngpcord.modify_on_device();
    ngpcord.sync_to_host();
    
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (const auto* b: bkts) {
      for (const auto node: *b) {
        const double* dpdx = stk::mesh::field_data(*dpdx_, node);
        const double* cord = stk::mesh::field_data(*coordinates_, node);
        int j=-1;
        if (cord[0] == 0) {
          if      (cord[2] ==  0 && (cord[1] == 0 || cord[1] == 20)) j = -1;
          else if (cord[2] == 20 && (cord[1] == 0 || cord[1] == 20)) j = -1;
          else if (cord[2] ==  0) j = 0;
          else if (cord[2] == 20) j = 1;
          else if (cord[1] ==  0) j = 2;
          else if (cord[1] == 20) j = 3;
          else                    j = 4;
        } else {
          if (0<cord[0]  && 0<cord[1]  && 0<cord[2] &&
              cord[0]<20 && cord[1]<20 && cord[2]<20)
                                  j = 5;
        }
        if (-1 < j) {
          for (int i=0; i<3; ++i)
            EXPECT_NEAR(dpdxref[j][i], dpdx[i], tol);
double diff = 0;
for (int i=0; i<3; ++i) diff += abs(dpdxref[j][i]-dpdx[i]);
std::cout
<<" diff:"<<diff
<<" cord:"
<<" "<<cord[0]
<<" "<<cord[1]
<<" "<<cord[2]
<<" dpdxref:"
<<" "<<dpdxref[j][0]
<<" "<<dpdxref[j][1]
<<" "<<dpdxref[j][2]
<<" dpdx:"
<<" "<<dpdx[0]
<<" "<<dpdx[1]
<<" "<<cord[2]
<<std::endl;
        }
      }
    }
  }
}
