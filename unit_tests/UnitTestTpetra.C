// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include <stk_util/parallel/Parallel.hpp>

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include "LinearSolvers.h"
#include "kernel/Kernel.h"
#include "kernel/KernelBuilder.h"
#include "SolverAlgorithmDriver.h"
#include "AssembleElemSolverAlgorithm.h"
#include "NaluEnv.h"
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
  sierra::nalu::EquationSystem* eqsys =
    realm.equationSystems_.equationSystemVector_[0];
  EXPECT_TRUE(eqsys != nullptr);
  sierra::nalu::LinearSystem* linsys = eqsys->linsys_;
  EXPECT_TRUE(linsys != nullptr);

  sierra::nalu::TpetraLinearSystem* tpetraLinsys =
    dynamic_cast<sierra::nalu::TpetraLinearSystem*>(linsys);
  ThrowRequireMsg(
    tpetraLinsys != nullptr, "Expected TpetraLinearSystem to be non-null");

  return tpetraLinsys;
}

sierra::nalu::AssembleElemSolverAlgorithm*
create_algorithm(sierra::nalu::Realm& realm, stk::mesh::Part& part)
{
  sierra::nalu::EquationSystem* eqsys =
    realm.equationSystems_.equationSystemVector_[0];
  EXPECT_TRUE(eqsys != nullptr);

  std::pair<sierra::nalu::AssembleElemSolverAlgorithm*, bool> solverAlgResult =
    sierra::nalu::build_or_add_part_to_solver_alg(
      *eqsys, part, eqsys->solverAlgDriver_->solverAlgorithmMap_);

  EXPECT_TRUE(solverAlgResult.second);
  ThrowRequireMsg(
    solverAlgResult.first != nullptr,
    "Error, failed to obtain non-null solver-algorithm object.");

  if (realm.geometryAlgDriver_ == nullptr) {
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
  sierra::nalu::EquationSystem* eqsys =
    realm.equationSystems_.equationSystemVector_[0];

  auto solverAlgMap = eqsys->solverAlgDriver_->solverAlgorithmMap_;
  EXPECT_EQ(1u, solverAlgMap.size());
  sierra::nalu::SolverAlgorithm* solverAlg = solverAlgMap.begin()->second;
  ThrowRequireMsg(solverAlg != nullptr, "Error, null solver-algorithm");

  sierra::nalu::AssembleElemSolverAlgorithm* assembleElemSolverAlgorithm =
    dynamic_cast<sierra::nalu::AssembleElemSolverAlgorithm*>(solverAlg);
  ThrowRequireMsg(
    assembleElemSolverAlgorithm != nullptr,
    "Error, failed to dynamic_cast to AssembleElemSolverAlgorithm.");
  return assembleElemSolverAlgorithm;
}

static const double elemVals[8][8] = {
  {2.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2},
  {-0.8, 2.0, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3},
  {-0.7, -0.8, 2.0, -0.8, -0.7, -0.6, -0.5, -0.4},
  {-0.6, -0.7, -0.8, 2.0, -0.8, -0.7, -0.6, -0.5},
  {-0.5, -0.6, -0.7, -0.8, 2.0, -0.8, -0.7, -0.6},
  {-0.4, -0.5, -0.6, -0.7, -0.8, 2.0, -0.8, -0.7},
  {-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, 2.0, -0.8},
  {-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, 2.0}};

// elem 1 nodes: 1 2 4 3 5 6 8 7
// elem 2 nodes: 5 6 8 7 9 10 12 11
// local elemVals coeffs above have been ordered appropriately into the
// following lhsVals table.
static const double lhsVals[12][12] = {
  {2.0, -0.8, -0.6, -0.7, -0.5, -0.4, -0.2, -0.3, 0.0, 0.0, 0.0, 0.0},
  {-0.8, 2.0, -0.7, -0.8, -0.6, -0.5, -0.3, -0.4, 0.0, 0.0, 0.0, 0.0},
  {-0.6, -0.7, 2.0, -0.8, -0.8, -0.7, -0.5, -0.6, 0.0, 0.0, 0.0, 0.0},
  {-0.7, -0.8, -0.8, 2.0, -0.7, -0.6, -0.4, -0.5, 0.0, 0.0, 0.0, 0.0},
  {-0.5, -0.6, -0.8, -0.7, 2.0 * 2, -0.8 * 2, -0.6 * 2, -0.7 * 2, -0.5, -0.4,
   -0.2, -0.3},
  {-0.4, -0.5, -0.7, -0.6, -0.8 * 2, 2.0 * 2, -0.7 * 2, -0.8 * 2, -0.6, -0.5,
   -0.3, -0.4},
  {-0.2, -0.3, -0.5, -0.4, -0.6 * 2, -0.7 * 2, 2.0 * 2, -0.8 * 2, -0.8, -0.7,
   -0.5, -0.6},
  {-0.3, -0.4, -0.6, -0.5, -0.7 * 2, -0.8 * 2, -0.8 * 2, 2.0 * 2, -0.7, -0.6,
   -0.4, -0.5},
  {0.0, 0.0, 0.0, 0.0, -0.5, -0.6, -0.8, -0.7, 2.0, -0.8, -0.6, -0.7},
  {0.0, 0.0, 0.0, 0.0, -0.4, -0.5, -0.7, -0.6, -0.8, 2.0, -0.7, -0.8},
  {0.0, 0.0, 0.0, 0.0, -0.2, -0.3, -0.5, -0.4, -0.6, -0.7, 2.0, -0.8},
  {0.0, 0.0, 0.0, 0.0, -0.3, -0.4, -0.6, -0.5, -0.7, -0.8, -0.8, 2.0}};

class TestKernel : public sierra::nalu::NGPKernel<TestKernel>
{
public:
  TestKernel(stk::topology elemTopo)
    : numNodesPerElem(elemTopo.num_nodes()),
      d_elemVals("device-elem-vals", 8, 8)
  {
    Kokkos::View<double**, sierra::nalu::MemSpace>::HostMirror h_elemVals =
      Kokkos::create_mirror_view(d_elemVals);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        h_elemVals(i, j) = elemVals[i][j];
      }
    }
    Kokkos::deep_copy(d_elemVals, h_elemVals);
  }

  KOKKOS_FUNCTION
  TestKernel() : numNodesPerElem(0), d_elemVals() {}

  KOKKOS_FUNCTION
  TestKernel(const TestKernel& src)
    : numNodesPerElem(src.numNodesPerElem), d_elemVals(src.d_elemVals)
  {
  }

  using sierra::nalu::Kernel::execute;
  KOKKOS_FUNCTION
  virtual void execute(
    sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem>& lhs,
    sierra::nalu::SharedMemView<DoubleType*, sierra::nalu::DeviceShmem>& rhs,
    sierra::nalu::ScratchViews<
      DoubleType,
      sierra::nalu::DeviceTeamHandleType,
      sierra::nalu::DeviceShmem>& /* scratchViews */)
  {
    const bool check0 = numNodesPerElem * numNodesPerElem == lhs.size();
    const bool check1 = numNodesPerElem == rhs.size();
    const bool check2 = numNodesPerElem == 8;
    if (check0 && check1 && check2) {
      for (unsigned i = 0; i < numNodesPerElem; ++i) {
        for (unsigned j = 0; j < numNodesPerElem; ++j) {
          lhs(i, j) = d_elemVals(i, j);
        }
      }
    }
  }

private:
  unsigned numNodesPerElem;
  Kokkos::View<double**, sierra::nalu::MemSpace> d_elemVals;
};

std::vector<unsigned>
get_gold_row_lengths(int numProcs, int localProc)
{
  std::vector<unsigned> goldRowLengths = {8,  8,  8, 8, 12, 12,
                                          12, 12, 8, 8, 8,  8};
  if (numProcs == 2) {
    if (localProc == 0) {
      goldRowLengths = {8, 8, 8, 8, 12, 12, 12, 12};
    } else {
      goldRowLengths = {8, 8, 8, 8};
    }
  }
  return goldRowLengths;
}

void
verify_graph_for_2_hex8_mesh(
  int numProcs, int localProc, sierra::nalu::TpetraLinearSystem* tpetraLinsys)
{
  unsigned expectedNumGlobalRows = 12;
  unsigned expectedNumOwnedRows = expectedNumGlobalRows;
  if (numProcs == 2) {
    expectedNumOwnedRows = localProc == 0 ? 8 : 4;
  }
  EXPECT_EQ(
    expectedNumGlobalRows, tpetraLinsys->getOwnedGraph()->getGlobalNumRows());
  EXPECT_EQ(
    expectedNumOwnedRows, tpetraLinsys->getOwnedGraph()->getLocalNumRows());

  std::vector<unsigned> goldRowLengths =
    get_gold_row_lengths(numProcs, localProc);

  for (unsigned localRow = 0; localRow < expectedNumOwnedRows; ++localRow) {
    EXPECT_EQ(
      goldRowLengths[localRow],
      tpetraLinsys->getOwnedGraph()->getNumEntriesInLocalRow(localRow))
      << "P" << localProc << ", localRow=" << localRow;
  }
}

void
verify_matrix_for_2_hex8_mesh(
  int numProcs, int localProc, sierra::nalu::TpetraLinearSystem* tpetraLinsys)
{
  Teuchos::RCP<sierra::nalu::LinSys::Matrix> ownedMatrix =
    tpetraLinsys->getOwnedMatrix();
  EXPECT_NE(nullptr, ownedMatrix.get());
  unsigned expectedGlobalNumRows = 12;
  int expectedLocalNumRows = 12;
  if (numProcs == 2) {
    expectedLocalNumRows = localProc == 0 ? 8 : 4;
  }
  EXPECT_EQ(expectedGlobalNumRows, ownedMatrix->getGlobalNumRows());
  EXPECT_EQ((unsigned)expectedLocalNumRows, ownedMatrix->getLocalNumRows());

  Teuchos::RCP<const sierra::nalu::LinSys::Map> rowMap =
    ownedMatrix->getRowMap();
  Teuchos::RCP<const sierra::nalu::LinSys::Map> colMap =
    ownedMatrix->getColMap();

  for (sierra::nalu::LinSys::LocalOrdinal rowlid = 0;
       rowlid < expectedLocalNumRows; ++rowlid) {
    sierra::nalu::LinSys::GlobalOrdinal rowgid =
      rowMap->getGlobalElement(rowlid);
    unsigned rowLength = ownedMatrix->getNumEntriesInGlobalRow(rowgid);
    sierra::nalu::LinSys::LocalIndicesHost inds;
    sierra::nalu::LinSys::LocalValuesHost vals;
    ownedMatrix->getLocalRowView(rowlid, inds, vals);
    for (unsigned j = 0; j < rowLength; ++j) {
      sierra::nalu::LinSys::GlobalOrdinal colgid =
        colMap->getGlobalElement(inds[j]);
      EXPECT_NEAR(lhsVals[rowgid - 1][colgid - 1], vals[j], 1.e-9)
        << "failed for row=" << rowgid << ",col=" << colgid;
    }
  }
}

TestKernel*
create_and_register_kernel(
  sierra::nalu::AssembleElemSolverAlgorithm* solverAlg, stk::topology elemTopo)
{
  TestKernel* testKernel = new TestKernel(elemTopo);
  solverAlg->dataNeededByKernels_.add_cvfem_volume_me(
    sierra::nalu::MasterElementRepo::get_volume_master_element(elemTopo));
  solverAlg->activeKernels_.push_back(testKernel);
  return testKernel;
}

sierra::nalu::Realm&
setup_realm(unit_test_utils::NaluTest& naluObj, const std::string& meshSpec)
{
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.setup_nodal_fields();

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  realm.timeIntegrator_ = &timeIntegrator;
  auto& part = realm.meta_data().declare_part("block_1");
  realm.register_nodal_fields(&part);
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());
  realm.set_global_id();

  // Reset it back to nullptr so that we don't carry around a stale pointer
  realm.timeIntegrator_ = nullptr;
  return realm;
}

sierra::nalu::Realm&
setup_solver_alg_and_linsys(
  unit_test_utils::NaluTest& naluObj, const std::string& meshSpec)
{
  sierra::nalu::Realm& realm = setup_realm(naluObj, meshSpec);
  stk::mesh::Part& block_1 = *realm.meta_data().get_part("block_1");
  sierra::nalu::AssembleElemSolverAlgorithm* solverAlg =
    create_algorithm(realm, block_1);
  create_and_register_kernel(solverAlg, block_1.topology());
  return realm;
}

TEST(Tpetra, basic)
{
  const int numProcs = stk::parallel_machine_size(MPI_COMM_WORLD);
  if (numProcs > 2) {
    GTEST_SKIP();
  }
  int localProc = stk::parallel_machine_rank(MPI_COMM_WORLD);

  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm =
    setup_solver_alg_and_linsys(naluObj, "generated:1x1x2");

  sierra::nalu::TpetraLinearSystem* tpetraLinsys =
    get_TpetraLinearSystem(naluObj);
  sierra::nalu::AssembleElemSolverAlgorithm* solverAlg =
    get_AssembleElemSolverAlgorithm(naluObj);

  tpetraLinsys->buildElemToNodeGraph(solverAlg->partVec_);
  tpetraLinsys->finalizeLinearSystem();

  verify_graph_for_2_hex8_mesh(numProcs, localProc, tpetraLinsys);

  auto meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element<
    sierra::nalu::AlgTraitsHex8>();
  auto& dataNeeded = solverAlg->dataNeededByKernels_;
  dataNeeded.add_cvfem_volume_me(meSCV);
  auto* coordsField = realm.meta_data().coordinate_field();
  dataNeeded.add_coordinates_field(
    *coordsField, 3, sierra::nalu::CURRENT_COORDINATES);
  dataNeeded.add_master_element_call(
    sierra::nalu::SCV_VOLUME, sierra::nalu::CURRENT_COORDINATES);

  solverAlg->execute();
  tpetraLinsys->loadComplete();
  verify_matrix_for_2_hex8_mesh(numProcs, localProc, tpetraLinsys);
}
