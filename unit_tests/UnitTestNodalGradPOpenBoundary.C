/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "memory"
#include "functional"

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"
#include "UnitTestUtils.h"

#include "AssembleNodalGradPOpenBoundaryAlgorithm.h"
#include "ComputeGeometryBoundaryAlgorithm.h"
#include "SolutionOptions.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

 
namespace sierra {
namespace nalu{

struct HelperObjectsNodalGradPOpenBoundary {
  HelperObjectsNodalGradPOpenBoundary(
    stk::mesh::BulkData& bulk, 
    stk::mesh::Part* part,
    const bool zeroGrad,
    const bool useShifted)
  : 
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(unit_test_utils::get_default_inputs())),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    NodalGradPOpenBoundaryAlg(),
    computeGeomBoundAlg()
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    realm.solutionOptions_->activateOpenMdotCorrection_ = true;
    realm.solutionOptions_->explicitlyZeroOpenPressureGradient_ = zeroGrad;
    NodalGradPOpenBoundaryAlg.reset(new AssembleNodalGradPOpenBoundaryAlgorithm(realm, part, useShifted));
    computeGeomBoundAlg.reset(new ComputeGeometryBoundaryAlgorithm(realm, part));
  }

  ~HelperObjectsNodalGradPOpenBoundary()
  {
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;
  }

  HelperObjectsNodalGradPOpenBoundary() = delete;
  HelperObjectsNodalGradPOpenBoundary(const HelperObjectsNodalGradPOpenBoundary&) = delete;

  const YAML::Node realmDefaultNode;
  std::unique_ptr<unit_test_utils::NaluTest> naluObj;
  sierra::nalu::Realm& realm;
  std::unique_ptr<AssembleNodalGradPOpenBoundaryAlgorithm> NodalGradPOpenBoundaryAlg;
  std::unique_ptr<ComputeGeometryBoundaryAlgorithm>        computeGeomBoundAlg;
};

#ifndef KOKKOS_ENABLE_CUDA
TEST_F(Hex8MeshWithNSOFields, nodal_grad_popen_boundary) {
  const int np  = bulk.parallel_size();
  const std::string meshSpec = "generated:1x1x" + std::to_string(np);
  fill_mesh_and_initialize_test_fields(meshSpec, true);
  stk::mesh::Part* surface1 = meta.get_part("surface_1");
  
  const double x = -0.125;
  const double y =  0.25 ;
  const stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& nodeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, all_local);
  std::function<void(bool,bool)> run_alg = [&](bool zeroGrad, bool useShifted) {
    stk::mesh::field_fill(0.0, *dpdx);
    HelperObjectsNodalGradPOpenBoundary helperObjs(bulk, surface1, zeroGrad, useShifted);
    helperObjs.computeGeomBoundAlg->execute();
    helperObjs.NodalGradPOpenBoundaryAlg->execute();
    for (const stk::mesh::Bucket* b : nodeBuckets)
    {
      for (stk::mesh::Entity node : *b)
      {
        const double* dp = stk::mesh::field_data(*dpdx, node);
        const double*  C = stk::mesh::field_data(*coordField, node);
        int I[3] = {0,0,0};
        for (int i=0; i<3; ++i) if (C[i]) I[i] = 1; // mesh boundary
        double re[3] = {x+y*I[0], x+y*I[1], x+y*I[2]};
        if (0 < C[2] && C[2] < np) re[2] = 0;       // mesh interior
        for (int i=0; i<3; ++i) 
          EXPECT_NEAR(re[i], dp[i], tol);
      }
    }
  };
  run_alg(false, false);
  run_alg(true, false);
  run_alg(false, true);
  run_alg(true, true);
}
#endif
}
}
