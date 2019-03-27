/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "gtest/gtest.h"
#include <stk_util/parallel/Parallel.hpp>

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include "LinearSolvers.h"
#include "kernel/KernelBuilder.h"
#include "SolverAlgorithmDriver.h"
#include "AssembleElemSolverAlgorithm.h"
#include "Realms.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "TpetraLinearSystem.h"
#include "SimdInterface.h"

#include <master_element/MasterElementFactory.h>
#include <string>

sierra::nalu::TpetraLinearSystem*
get_TpetraLinearSystem(unit_test_utils::NaluTest& naluObj)
{
  EXPECT_NE(nullptr, naluObj.sim_.realms_);
  EXPECT_FALSE(naluObj.sim_.realms_->realmVector_.empty());
  sierra::nalu::Realm& realm = *naluObj.sim_.realms_->realmVector_[0];
  EXPECT_FALSE(realm.equationSystems_.equationSystemVector_.empty());
  sierra::nalu::EquationSystem* eqsys = realm.equationSystems_.equationSystemVector_[0];
  EXPECT_TRUE(eqsys != nullptr);
  sierra::nalu::LinearSystem* linsys = eqsys->linsys_;
  EXPECT_TRUE(linsys != nullptr);

  sierra::nalu::TpetraLinearSystem* tpetraLinsys = dynamic_cast<sierra::nalu::TpetraLinearSystem*>(linsys);
  ThrowRequireMsg(tpetraLinsys != nullptr, "Expected TpetraLinearSystem to be non-null");

  return tpetraLinsys;
}

sierra::nalu::AssembleElemSolverAlgorithm*
create_algorithm(sierra::nalu::Realm& realm, stk::mesh::Part& part)
{
  sierra::nalu::EquationSystem* eqsys = realm.equationSystems_.equationSystemVector_[0];
  EXPECT_TRUE(eqsys != nullptr);

  std::pair<sierra::nalu::AssembleElemSolverAlgorithm*,bool> solverAlgResult =
      sierra::nalu::build_or_add_part_to_solver_alg(*eqsys, part, eqsys->solverAlgDriver_->solverAlgorithmMap_);

  EXPECT_TRUE(solverAlgResult.second);
  ThrowRequireMsg(solverAlgResult.first != nullptr,"Error, failed to obtain non-null solver-algorithm object.");

  if (realm.computeGeometryAlgDriver_ == nullptr) {
    realm.breadboard();
  }
  realm.register_interior_algorithm(&part);

  return solverAlgResult.first;
}

sierra::nalu::AssembleElemSolverAlgorithm*
get_AssembleElemSolverAlgorithm(unit_test_utils::NaluTest& naluObj)
{
  EXPECT_NE(nullptr, naluObj.sim_.realms_);
  EXPECT_FALSE(naluObj.sim_.realms_->realmVector_.empty());
  sierra::nalu::Realm& realm = *naluObj.sim_.realms_->realmVector_[0];
  EXPECT_FALSE(realm.equationSystems_.equationSystemVector_.empty());
  sierra::nalu::EquationSystem* eqsys = realm.equationSystems_.equationSystemVector_[0];

  auto solverAlgMap = eqsys->solverAlgDriver_->solverAlgorithmMap_;
  EXPECT_EQ(1u, solverAlgMap.size());
  sierra::nalu::SolverAlgorithm* solverAlg = solverAlgMap.begin()->second;
  ThrowRequireMsg(solverAlg != nullptr, "Error, null solver-algorithm");

  sierra::nalu::AssembleElemSolverAlgorithm* assembleElemSolverAlgorithm =
      dynamic_cast<sierra::nalu::AssembleElemSolverAlgorithm*>(solverAlg);
  ThrowRequireMsg(assembleElemSolverAlgorithm != nullptr, "Error, failed to dynamic_cast to AssembleElemSolverAlgorithm.");
  return assembleElemSolverAlgorithm;
}

static const double elemVals[8][8] = {
    {2.0, -0.8, -0.7, -0.6,  -0.5, -0.4, -0.3, -0.2},
    {-0.8, 2.0, -0.8, -0.7,  -0.6, -0.5, -0.4, -0.3},
    {-0.7, -0.8, 2.0, -0.8,  -0.7, -0.6, -0.5, -0.4},
    {-0.6, -0.7, -0.8, 2.0,  -0.8, -0.7, -0.6, -0.5},
    {-0.5, -0.6, -0.7, -0.8,  2.0, -0.8, -0.7, -0.6},
    {-0.4, -0.5, -0.6, -0.7, -0.8,  2.0, -0.8, -0.7},
    {-0.3, -0.4, -0.5, -0.6, -0.7, -0.8,  2.0, -0.8},
    {-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8,  2.0}
};

//elem 1 nodes: 1 2 4 3 5 6 8 7
//elem 2 nodes: 5 6 8 7 9 10 12 11
//local elemVals coeffs above have been ordered appropriately into the following
//lhsVals table.
static const double lhsVals[12][12] = {
    {2.0, -0.8, -0.6, -0.7,  -0.5,   -0.4,  -0.2,   -0.3,     0.0,  0.0,  0.0,  0.0},
    {-0.8, 2.0, -0.7, -0.8,  -0.6,   -0.5,  -0.3,   -0.4,     0.0,  0.0,  0.0,  0.0},
    {-0.6, -0.7, 2.0,  -0.8, -0.8,   -0.7,  -0.5,   -0.6,     0.0,  0.0,  0.0,  0.0},
    {-0.7, -0.8,-0.8,  2.0,  -0.7,   -0.6,  -0.4,   -0.5,     0.0,  0.0,  0.0,  0.0},
    {-0.5, -0.6, -0.8, -0.7,  2.0*2, -0.8*2,-0.6*2, -0.7*2,  -0.5, -0.4, -0.2, -0.3},
    {-0.4, -0.5, -0.7, -0.6, -0.8*2,  2.0*2,-0.7*2, -0.8*2,  -0.6, -0.5, -0.3, -0.4},
    {-0.2, -0.3, -0.5, -0.4, -0.6*2, -0.7*2, 2.0*2, -0.8*2,  -0.8, -0.7, -0.5, -0.6},
    {-0.3, -0.4, -0.6, -0.5, -0.7*2, -0.8*2,-0.8*2,  2.0*2,  -0.7, -0.6, -0.4, -0.5},
    {0.0,   0.0,  0.0,  0.0, -0.5,   -0.6,  -0.8,   -0.7,     2.0, -0.8, -0.6, -0.7},
    {0.0,   0.0,  0.0,  0.0, -0.4,   -0.5,  -0.7,   -0.6,    -0.8,  2.0, -0.7, -0.8},
    {0.0,   0.0,  0.0,  0.0, -0.2,   -0.3,  -0.5,   -0.4,    -0.6, -0.7,  2.0, -0.8},
    {0.0,   0.0,  0.0,  0.0, -0.3,   -0.4,  -0.6,   -0.5,    -0.7, -0.8, -0.8,  2.0}
};

class TestKernel : public sierra::nalu::Kernel {
public:
  TestKernel(stk::topology elemTopo)
    : numCallsToExecute(0), numNodesPerElem(elemTopo.num_nodes())
  {
  }

  using sierra::nalu::Kernel::execute;
  virtual void execute(
    sierra::nalu::SharedMemView<DoubleType**> &lhs,
    sierra::nalu::SharedMemView<DoubleType*> &rhs,
    sierra::nalu::ScratchViews<DoubleType> & /* scratchViews */)
  {
    EXPECT_EQ(numNodesPerElem*numNodesPerElem, lhs.size());
    EXPECT_EQ(numNodesPerElem, rhs.size());
    ThrowRequireMsg(numNodesPerElem == 8,"For now, this is hard-wired for hex-8.");

    for(unsigned i=0; i<numNodesPerElem; ++i) {
      for(unsigned j=0; j<numNodesPerElem; ++j) {
        lhs(i,j) = elemVals[i][j];
      }
    }

    ++numCallsToExecute;
  }

  unsigned numCallsToExecute;
private:
  unsigned numNodesPerElem;
};

std::vector<unsigned> get_gold_row_lengths(int numProcs, int localProc)
{
  std::vector<unsigned> goldRowLengths = {8, 8, 8, 8, 12, 12, 12, 12, 8, 8, 8, 8};
  if (numProcs == 2) {
    if (localProc==0) {
      goldRowLengths = {8, 8, 8, 8, 12, 12, 12, 12};
    }
    else {
      goldRowLengths = {8, 8, 8, 8};
    }
  }
  return goldRowLengths;
}

void verify_graph_for_2_hex8_mesh(int numProcs, int localProc, sierra::nalu::TpetraLinearSystem* tpetraLinsys)
{
  unsigned expectedNumGlobalRows = 12;
  unsigned expectedNumOwnedRows = expectedNumGlobalRows;
  if (numProcs == 2) {
    expectedNumOwnedRows = localProc==0 ? 8 : 4;
  }
  EXPECT_EQ(expectedNumGlobalRows, tpetraLinsys->getOwnedGraph()->getGlobalNumRows());
  EXPECT_EQ(expectedNumOwnedRows, tpetraLinsys->getOwnedGraph()->getNodeNumRows());

  std::vector<unsigned> goldRowLengths = get_gold_row_lengths(numProcs, localProc);

  for(unsigned localRow=0; localRow<expectedNumOwnedRows; ++localRow) {
    EXPECT_EQ(goldRowLengths[localRow], tpetraLinsys->getOwnedGraph()->getNumEntriesInLocalRow(localRow)) << "P"<<localProc<<", localRow="<<localRow;
  }
}

void verify_matrix_for_2_hex8_mesh(int numProcs, int localProc, sierra::nalu::TpetraLinearSystem* tpetraLinsys)
{
  Teuchos::RCP<sierra::nalu::LinSys::Matrix> ownedMatrix = tpetraLinsys->getOwnedMatrix();
  EXPECT_NE(nullptr, ownedMatrix.get());
  unsigned expectedGlobalNumRows = 12;
  int expectedLocalNumRows = 12;
  if (numProcs==2) {
    expectedLocalNumRows = localProc==0 ? 8 : 4;
  }
  EXPECT_EQ(expectedGlobalNumRows, ownedMatrix->getGlobalNumRows());
  EXPECT_EQ((unsigned)expectedLocalNumRows, ownedMatrix->getNodeNumRows());

  Teuchos::RCP<const sierra::nalu::LinSys::Map> rowMap = ownedMatrix->getRowMap();
  Teuchos::RCP<const sierra::nalu::LinSys::Map> colMap = ownedMatrix->getColMap();

  for(sierra::nalu::LinSys::LocalOrdinal rowlid=0; rowlid<expectedLocalNumRows; ++rowlid) {
    sierra::nalu::LinSys::GlobalOrdinal rowgid = rowMap->getGlobalElement(rowlid);
    unsigned rowLength = ownedMatrix->getNumEntriesInGlobalRow(rowgid);
    Teuchos::ArrayView<const sierra::nalu::LinSys::LocalOrdinal> inds;
    Teuchos::ArrayView<const double> vals;
    ownedMatrix->getLocalRowView(rowlid, inds, vals);
    for(unsigned j=0; j<rowLength; ++j) {
      sierra::nalu::LinSys::GlobalOrdinal colgid = colMap->getGlobalElement(inds[j]);
      EXPECT_NEAR(lhsVals[rowgid-1][colgid-1], vals[j], 1.e-9)<<"failed for row="<<rowgid<<",col="<<colgid;
    }
  }
}

TestKernel* create_and_register_kernel(sierra::nalu::AssembleElemSolverAlgorithm* solverAlg, stk::topology elemTopo)
{
  TestKernel* testKernel = new TestKernel(elemTopo);
  solverAlg->dataNeededByKernels_.add_cvfem_volume_me(sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo));
  solverAlg->activeKernels_.push_back(testKernel);
  return testKernel;
}

sierra::nalu::Realm& setup_realm(unit_test_utils::NaluTest& naluObj, const std::string& meshSpec)
{
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.setup_nodal_fields();
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());
  realm.set_global_id();
  return realm;
}

void setup_solver_alg_and_linsys(unit_test_utils::NaluTest& naluObj, const std::string& meshSpec)
{
  sierra::nalu::Realm& realm = setup_realm(naluObj, meshSpec);
  stk::mesh::Part& block_1 = *realm.meta_data().get_part("block_1");
  sierra::nalu::AssembleElemSolverAlgorithm* solverAlg = create_algorithm(realm, block_1);
  create_and_register_kernel(solverAlg, block_1.topology());
}

TEST(Tpetra, basic)
{
  int numProcs = stk::parallel_machine_size(MPI_COMM_WORLD);
  if (numProcs > 2) { return; }
  int localProc = stk::parallel_machine_rank(MPI_COMM_WORLD);

  unit_test_utils::NaluTest naluObj;
  setup_solver_alg_and_linsys(naluObj, "generated:1x1x2");

  sierra::nalu::TpetraLinearSystem* tpetraLinsys = get_TpetraLinearSystem(naluObj);
  sierra::nalu::AssembleElemSolverAlgorithm* solverAlg = get_AssembleElemSolverAlgorithm(naluObj);

  tpetraLinsys->buildElemToNodeGraph(solverAlg->partVec_);
  tpetraLinsys->finalizeLinearSystem();

  verify_graph_for_2_hex8_mesh(numProcs, localProc, tpetraLinsys);

  solverAlg->execute();
  tpetraLinsys->loadComplete();

  verify_matrix_for_2_hex8_mesh(numProcs, localProc, tpetraLinsys);
}
