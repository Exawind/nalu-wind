
#include "memory"

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"
#include "UnitTestUtils.h"

#include "ABLProfileFunction.h"
#include "AssembleElemSolverAlgorithm.h"
#include "AssembleNodalGradPOpenBoundaryAlgorithm.h"
#include "ComputeGeometryBoundaryAlgorithm.h"
#include "EquationSystem.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/SkinBoundary.hpp>
#include <stk_topology/topology.hpp>

 
namespace sierra {
namespace nalu{

struct HelperObjectsNodalGradPOpenBoundary {
  HelperObjectsNodalGradPOpenBoundary(
    stk::mesh::BulkData& bulk, 
    stk::mesh::Part* part)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    NodalGradPOpenBoundaryAlg(),
    computeGeomBoundAlg()
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    realm.solutionOptions_->activateOpenMdotCorrection_ = true;
    NodalGradPOpenBoundaryAlg.reset(new AssembleNodalGradPOpenBoundaryAlgorithm(realm, part, false));
    computeGeomBoundAlg.reset(new ComputeGeometryBoundaryAlgorithm(realm, part));
  }

  ~HelperObjectsNodalGradPOpenBoundary()
  {
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;
  }

  HelperObjectsNodalGradPOpenBoundary() = delete;
  HelperObjectsNodalGradPOpenBoundary(const HelperObjectsNodalGradPOpenBoundary&) = delete;

  const YAML::Node yamlNode;
  const YAML::Node realmDefaultNode;
  std::unique_ptr<unit_test_utils::NaluTest> naluObj;
  sierra::nalu::Realm& realm;
  std::unique_ptr<AssembleNodalGradPOpenBoundaryAlgorithm> NodalGradPOpenBoundaryAlg;
  std::unique_ptr<ComputeGeometryBoundaryAlgorithm> computeGeomBoundAlg;
};

#ifndef KOKKOS_ENABLE_CUDA
TEST_F(Hex8MeshWithNSOFields, nodal_grad_popen_boundary) {
  const std::string meshSpec = "generated:20x20x20";
  fill_mesh_and_initialize_test_fields(meshSpec, true);
  stk::mesh::Part* surface1 = meta.get_part("surface_1");
  
  const double dpdxref[6][3] = {{-0.25, 0.0,  -0.25},
                                {-0.25, 0.0,   0.25},
                                {-0.25,-0.25,  0.0},
                                {-0.25, 0.25,  0.0},
                                {-0.5,  0.0,   0.0},
                                { 0.0,  0.0,   0.0}};

  HelperObjectsNodalGradPOpenBoundary helperObjs(bulk, surface1);
  helperObjs.computeGeomBoundAlg->execute();

  stk::mesh::field_fill(0.0, *dpdx);
  helperObjs.NodalGradPOpenBoundaryAlg->execute();

  const stk::mesh::Selector all_local = meta.universal_part() & meta.locally_owned_part();
  const stk::mesh::BucketVector& nodeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, all_local);
  for (const stk::mesh::Bucket* b : nodeBuckets)
  {
    for (stk::mesh::Entity node : *b)
    {
      const double* dpdxData = stk::mesh::field_data(*dpdx, node);
      const double* cordData = stk::mesh::field_data(*coordField, node);
      int j=-1;
      if (cordData[0] == 0) {
        if      (cordData[2] ==  0 && (cordData[1] == 0 || cordData[1] == 20)) j = -1;
        else if (cordData[2] == 20 && (cordData[1] == 0 || cordData[1] == 20)) j = -1;
        else if (cordData[2] ==  0) j = 0;
        else if (cordData[2] == 20) j = 1;
        else if (cordData[1] ==  0) j = 2;
        else if (cordData[1] == 20) j = 3;
        else                        j = 4;
        if (stk::parallel_machine_size(comm)==2 && cordData[2] == 10) j = -1;
      } else {
        if (0<cordData[0]  && 0<cordData[1]  && 0<cordData[2] &&
            cordData[0]<20 && cordData[1]<20 && cordData[2]<20)
                                    j = 5;
      }
      if (-1 < j) {
        for (int i=0; i<3; ++i) 
          EXPECT_NEAR(dpdxref[j][i], dpdxData[i], tol);
      }
    }
  }
}

#endif

}
}
