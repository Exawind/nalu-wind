#ifndef _UnitTestHelperObjects_h_
#define _UnitTestHelperObjects_h_

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"

#include "AssembleElemSolverAlgorithm.h"
#include "AssembleFaceElemSolverAlgorithm.h"
#include "EquationSystem.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace unit_test_utils {

struct HelperObjects {
  HelperObjects(stk::mesh::BulkData& bulk, stk::topology topo, int numDof, stk::mesh::Part* part)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    eqSystems(realm),
    eqSystem(eqSystems),
    linsys(new unit_test_utils::TestLinearSystem(realm, numDof, &eqSystem)),
    assembleElemSolverAlg(nullptr)
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    eqSystem.linsys_ = linsys;
    assembleElemSolverAlg = new sierra::nalu::AssembleElemSolverAlgorithm(realm, part, &eqSystem, topo.rank(), topo.num_nodes());
  }

  ~HelperObjects()
  {
    assembleElemSolverAlg->activeKernels_.clear();
    delete assembleElemSolverAlg;
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;

    delete naluObj;
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  unit_test_utils::NaluTest* naluObj;
  sierra::nalu::Realm& realm;
  sierra::nalu::EquationSystems eqSystems;
  sierra::nalu::EquationSystem eqSystem;
  unit_test_utils::TestLinearSystem* linsys;
  sierra::nalu::AssembleElemSolverAlgorithm* assembleElemSolverAlg;
};


struct FaceElemHelperObjects : HelperObjects {
  FaceElemHelperObjects(stk::mesh::BulkData& bulk, stk::topology faceTopo, stk::topology elemTopo, int numDof, stk::mesh::Part* part)
  : HelperObjects(bulk, elemTopo, numDof, part)
  {
    assembleFaceElemSolverAlg = new sierra::nalu::AssembleFaceElemSolverAlgorithm(realm, part, &eqSystem, faceTopo.num_nodes(), elemTopo.num_nodes());
  }

  ~FaceElemHelperObjects() {    delete assembleFaceElemSolverAlg;
  }

  sierra::nalu::AssembleFaceElemSolverAlgorithm* assembleFaceElemSolverAlg;
};

}

#endif

