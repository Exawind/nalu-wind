#ifndef _UnitTestHelperObjects_h_
#define _UnitTestHelperObjects_h_

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"

#include "AssembleElemSolverAlgorithm.h"
#include "AssembleFaceElemSolverAlgorithm.h"
#include "EquationSystem.h"
#include "kernel/Kernel.h"

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
    linsys(new unit_test_utils::TestLinearSystem(realm, numDof, &eqSystem, topo)),
    assembleElemSolverAlg(nullptr)
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    eqSystem.linsys_ = linsys;
    assembleElemSolverAlg = new sierra::nalu::AssembleElemSolverAlgorithm(realm, part, &eqSystem, topo.rank(), topo.num_nodes());
  }

  virtual ~HelperObjects()
  {
    delete assembleElemSolverAlg;
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;

    delete naluObj;
  }

  virtual void execute()
  {
    assembleElemSolverAlg->execute();
    for (auto kern: assembleElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleElemSolverAlg->activeKernels_.clear();

    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  void print_lhs_and_rhs() const
  {
    auto oldPrec = std::cerr.precision();
    std::cerr.precision(14);
    for(unsigned i=0; i<linsys->lhs_.extent(0); ++i) {
      std::cerr<<"{";
      for(unsigned j=0; j<linsys->lhs_.extent(1); ++j) {
        std::cerr<< linsys->lhs_(i,j)<<", ";
      }
      std::cerr<<"}"<<std::endl;
    }
    std::cerr<<"rhs: {";
    for(unsigned i=0; i<linsys->lhs_.extent(0); ++i) {
      std::cerr<< linsys->rhs_(i)<<", ";
    }
    std::cerr<<"}"<<std::endl;
    std::cerr.precision(oldPrec);
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

  virtual ~FaceElemHelperObjects()
  {
    delete assembleFaceElemSolverAlg;
  }

  virtual void execute() override
  {
    assembleFaceElemSolverAlg->execute();
    for (auto kern: assembleFaceElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleFaceElemSolverAlg->activeKernels_.clear();
  }

  sierra::nalu::AssembleFaceElemSolverAlgorithm* assembleFaceElemSolverAlg;
};

}

#endif

