#ifndef _UnitTestTpetraHelperObjects_h_
#define _UnitTestTpetraHelperObjects_h_

#include "UnitTestRealm.h"

#include "AssembleElemSolverAlgorithm.h"
#include "AssembleFaceElemSolverAlgorithm.h"
#include "TpetraLinearSystem.h"
#include "EquationSystem.h"
#include "kernel/Kernel.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

namespace unit_test_utils {

struct TpetraHelperObjectsBase {
  TpetraHelperObjectsBase(stk::mesh::BulkData& bulk, int numDof)
  : yamlNode(unit_test_utils::get_default_inputs()),
    realmDefaultNode(unit_test_utils::get_realm_default_node()),
    naluObj(new unit_test_utils::NaluTest(yamlNode)),
    realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
    eqSystems(realm),
    eqSystem(eqSystems),
    linsys(new sierra::nalu::TpetraLinearSystem(realm, numDof, &eqSystem, nullptr))
  {
    realm.metaData_ = &bulk.mesh_meta_data();
    realm.bulkData_ = &bulk;
    eqSystem.linsys_ = linsys;
  }

  virtual ~TpetraHelperObjectsBase()
  {
    realm.metaData_ = nullptr;
    realm.bulkData_ = nullptr;

    delete naluObj;
  }

  virtual void execute() { }

  template<typename LHSType, typename RHSType>
  void check_against_gold_values(unsigned rhsSize, const LHSType& lhs, const RHSType& rhs)
  {
    using MatrixType = sierra::nalu::LinSys::LocalMatrix;
    const MatrixType& localMatrix = linsys->getOwnedMatrix()->getLocalMatrix();
  
    using VectorType = sierra::nalu::LinSys::LocalVector;
    const VectorType& localRhs = linsys->getOwnedRhs()->getLocalView<sierra::nalu::DeviceSpace>();
  
    EXPECT_EQ(rhsSize, localMatrix.numRows());
    EXPECT_EQ(rhsSize, localRhs.size());
  
    stk::mesh::Entity elem = realm.bulkData_->get_entity(stk::topology::ELEM_RANK, 1);
    const stk::mesh::Entity* elemNodes = realm.bulkData_->begin_nodes(elem);
    unsigned numElemNodes = realm.bulkData_->num_nodes(elem);
    EXPECT_EQ(rhsSize, numElemNodes*linsys->numDof());
  
    for(unsigned i=0; i<numElemNodes; ++i) {
      int rowId = linsys->getRowLID(elemNodes[i]);
      for(unsigned d=0; d<linsys->numDof(); ++d) {
        KokkosSparse::SparseRowViewConst<MatrixType> constRowView = localMatrix.rowConst(rowId+d);
        EXPECT_EQ(rhsSize, constRowView.length);
  
        for(unsigned j=0; j<numElemNodes; ++j) {
          int colId = linsys->getColLID(elemNodes[j]);
          for(unsigned dd=0; dd<linsys->numDof(); ++dd) {
            EXPECT_NEAR(lhs[i][j], constRowView.value(colId+dd), 1.e-14);
          }
        }
  
        EXPECT_NEAR(rhs[i], localRhs(rowId+d,0), 1.e-14);
      }
    }
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  unit_test_utils::NaluTest* naluObj;
  sierra::nalu::Realm& realm;
  sierra::nalu::EquationSystems eqSystems;
  sierra::nalu::EquationSystem eqSystem;
  sierra::nalu::TpetraLinearSystem* linsys;
};

struct TpetraHelperObjectsElem : public TpetraHelperObjectsBase {
  TpetraHelperObjectsElem(stk::mesh::BulkData& bulk, stk::topology topo, int numDof, stk::mesh::Part* part)
  : TpetraHelperObjectsBase(bulk, numDof),
    assembleElemSolverAlg(new sierra::nalu::AssembleElemSolverAlgorithm(realm, part, &eqSystem, topo.rank(), topo.num_nodes()))
  {
  }

  virtual ~TpetraHelperObjectsElem()
  {
    delete assembleElemSolverAlg;
  }

  virtual void execute()
  {
    linsys->buildElemToNodeGraph({&realm.metaData_->universal_part()});
    linsys->finalizeLinearSystem();

    assembleElemSolverAlg->execute();
    for (auto kern: assembleElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleElemSolverAlg->activeKernels_.clear();
  }

  sierra::nalu::AssembleElemSolverAlgorithm* assembleElemSolverAlg;
};


struct TpetraHelperObjectsFaceElem : public TpetraHelperObjectsBase {
  TpetraHelperObjectsFaceElem(stk::mesh::BulkData& bulk, stk::topology faceTopo, stk::topology elemTopo, int numDof, stk::mesh::Part* part)
  : TpetraHelperObjectsBase(bulk, numDof),
    assembleFaceElemSolverAlg(new sierra::nalu::AssembleFaceElemSolverAlgorithm(realm, part, &eqSystem, faceTopo.num_nodes(), elemTopo.num_nodes()))
  {
  }

  virtual ~TpetraHelperObjectsFaceElem()
  {
    delete assembleFaceElemSolverAlg;
  }

  virtual void execute() override
  {
    linsys->buildElemToNodeGraph({&realm.metaData_->universal_part()});
    linsys->finalizeLinearSystem();

    assembleFaceElemSolverAlg->execute();
    for (auto kern: assembleFaceElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleFaceElemSolverAlg->activeKernels_.clear();
  }

  sierra::nalu::AssembleFaceElemSolverAlgorithm* assembleFaceElemSolverAlg;
};

}

#endif

