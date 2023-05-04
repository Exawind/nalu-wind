#ifndef _UnitTestHelperObjects_h_
#define _UnitTestHelperObjects_h_

#include "UnitTestRealm.h"
#include "UnitTestLinearSystem.h"

#include "AssembleEdgeSolverAlgorithm.h"
#include "AssembleElemSolverAlgorithm.h"
#include "AssembleFaceElemSolverAlgorithm.h"
#include "AssembleNGPNodeSolverAlgorithm.h"
#include "edge_kernels/AssembleAMSEdgeKernelAlg.h"
#include "EquationSystem.h"
#include "kernel/Kernel.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_topology/topology.hpp>

#include <memory>
#include <FieldManager.h>

namespace unit_test_utils {

struct HelperObjectsBase
{
  HelperObjectsBase(
    std::shared_ptr<stk::mesh::BulkData> bulk,
    YAML::Node yaml_node = unit_test_utils::get_default_inputs(),
    YAML::Node realm_node = unit_test_utils::get_realm_default_node())
    : yamlNode(yaml_node),
      realmDefaultNode(realm_node),
      naluObj(new unit_test_utils::NaluTest(yamlNode)),
      realm(naluObj->create_realm(realmDefaultNode, "multi_physics", false)),
      eqSystems(realm),
      eqSystem(eqSystems)
  {
    realm.bulkData_ = bulk;
    // hack
    // realm.setup_field_manager();
    const int numStates = 3;
    realm.fieldManager_ = std::make_unique<sierra::nalu::FieldManager>(realm.meta_data(), numStates);
  }

  virtual ~HelperObjectsBase() { delete naluObj; }

  virtual void execute() = 0;

  void print_lhs_and_rhs(const TestLinearSystem* linsys) const
  {
    auto oldPrec = std::cerr.precision();
    std::cerr.precision(14);
    std::cerr << "lhs:\n{" << std::endl;
    for (unsigned i = 0; i < linsys->lhs_.extent(0); ++i) {
      std::cerr << "{";
      for (unsigned j = 0; j < linsys->lhs_.extent(1); ++j) {
        std::cerr << linsys->lhs_(i, j) << ", ";
      }
      std::cerr << "};" << std::endl;
    }
    std::cerr << "};" << std::endl;
    std::cerr << "rhs:\n{";
    for (unsigned i = 0; i < linsys->lhs_.extent(0); ++i) {
      std::cerr << linsys->rhs_(i) << ", ";
    }
    std::cerr << "};" << std::endl;
    std::cerr.precision(oldPrec);
  }

  YAML::Node yamlNode;
  YAML::Node realmDefaultNode;
  unit_test_utils::NaluTest* naluObj;
  sierra::nalu::Realm& realm;
  sierra::nalu::EquationSystems eqSystems;
  sierra::nalu::EquationSystem eqSystem;
};

struct HelperObjects : public HelperObjectsBase
{
  HelperObjects(
    std::shared_ptr<stk::mesh::BulkData> bulk,
    stk::topology topo,
    int numDof,
    stk::mesh::Part* part,
    bool isEdge = false,
    YAML::Node yaml_node_pre_realm = unit_test_utils::get_default_inputs(),
    YAML::Node yaml_node_realm = unit_test_utils::get_realm_default_node())
    : HelperObjectsBase(bulk, yaml_node_pre_realm, yaml_node_realm),
      linsys(new unit_test_utils::TestLinearSystem(
        realm, numDof, &eqSystem, topo, isEdge))
  {
    eqSystem.linsys_ = linsys;
    assembleElemSolverAlg = new sierra::nalu::AssembleElemSolverAlgorithm(
      realm, part, &eqSystem, topo.rank(), topo.num_nodes());
  }

  virtual ~HelperObjects() { delete assembleElemSolverAlg; }

  template <typename LHSType, typename RHSType>
  void check_against_gold_values(
    unsigned rhsSize, const LHSType& lhs, const RHSType& rhs)
  {
    EXPECT_EQ(rhsSize, linsys->hostlhs_.extent(0));
    EXPECT_EQ(rhsSize, linsys->hostrhs_.extent(0));

    stk::mesh::Entity elem =
      realm.bulkData_->get_entity(stk::topology::ELEM_RANK, 1);
    const stk::mesh::Entity* elemNodes = realm.bulkData_->begin_nodes(elem);
    unsigned numElemNodes = realm.bulkData_->num_nodes(elem);
    unsigned nDof = linsys->numDof();
    EXPECT_EQ(rhsSize, numElemNodes * nDof);

    for (unsigned i = 0; i < numElemNodes; ++i) {
      unsigned rowId = linsys->getRowLID(elemNodes[i]);
      for (unsigned d = 0; d < nDof; ++d) {
        unsigned goldRow = i * nDof + d;
        unsigned linsysRow = rowId * nDof + d;

        for (unsigned j = 0; j < numElemNodes; ++j) {
          unsigned colId = linsys->getColLID(elemNodes[j]);
          for (unsigned dd = 0; dd < nDof; ++dd) {
            unsigned goldCol = j * nDof + dd;
            unsigned linsysCol = colId * nDof + dd;

            EXPECT_NEAR(
              lhs[goldRow][goldCol], linsys->hostlhs_(linsysRow, linsysCol),
              1.e-14);
          }
        }

        EXPECT_NEAR(rhs[goldRow], linsys->hostrhs_(linsysRow), 1.e-14);
      }
    }
  }

  virtual void execute() override
  {
    assembleElemSolverAlg->execute();
    for (auto kern : assembleElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleElemSolverAlg->activeKernels_.clear();

    Kokkos::deep_copy(linsys->hostNumSumIntoCalls_, linsys->numSumIntoCalls_);
    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  void print_lhs_and_rhs() const
  {
    HelperObjectsBase::print_lhs_and_rhs(linsys);
  }

  unit_test_utils::TestLinearSystem* linsys{nullptr};
  sierra::nalu::AssembleElemSolverAlgorithm* assembleElemSolverAlg{nullptr};
};

struct FaceElemHelperObjects : HelperObjects
{
  FaceElemHelperObjects(
    std::shared_ptr<stk::mesh::BulkData> bulk,
    stk::topology faceTopo,
    stk::topology elemTopo,
    int numDof,
    stk::mesh::Part* part,
    bool isEdge = false)
    : HelperObjects(bulk, elemTopo, numDof, part, isEdge)
  {
    assembleFaceElemSolverAlg =
      new sierra::nalu::AssembleFaceElemSolverAlgorithm(
        realm, part, &eqSystem, faceTopo.num_nodes(), elemTopo.num_nodes());
  }

  virtual ~FaceElemHelperObjects() { delete assembleFaceElemSolverAlg; }

  virtual void execute() override
  {
    assembleFaceElemSolverAlg->execute();
    for (auto kern : assembleFaceElemSolverAlg->activeKernels_)
      kern->free_on_device();
    assembleFaceElemSolverAlg->activeKernels_.clear();

    Kokkos::deep_copy(linsys->hostNumSumIntoCalls_, linsys->numSumIntoCalls_);
    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  sierra::nalu::AssembleFaceElemSolverAlgorithm* assembleFaceElemSolverAlg;
};

struct EdgeHelperObjects : public HelperObjectsBase
{
  EdgeHelperObjects(
    std::shared_ptr<stk::mesh::BulkData> bulk, stk::topology topo, int numDof)
    : HelperObjectsBase(bulk),
      linsys(new TestEdgeLinearSystem(realm, numDof, &eqSystem, topo))
  {
    eqSystem.linsys_ = linsys;
  }

  virtual ~EdgeHelperObjects()
  {
    if (edgeAlg != nullptr)
      delete edgeAlg;
  }

  template <typename T, class... Args>
  void create(stk::mesh::Part* part, Args&&... args)
  {
    ThrowRequire(edgeAlg == nullptr);
    edgeAlg = new T(realm, part, &eqSystem, std::forward<Args>(args)...);
  }

  virtual void execute() override
  {
    ThrowRequire(edgeAlg != nullptr);
    edgeAlg->execute();

    Kokkos::deep_copy(linsys->hostNumSumIntoCalls_, linsys->numSumIntoCalls_);
    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  void print_lhs_and_rhs() const
  {
    HelperObjectsBase::print_lhs_and_rhs(linsys);
  }

  unit_test_utils::TestEdgeLinearSystem* linsys{nullptr};
  sierra::nalu::AssembleEdgeSolverAlgorithm* edgeAlg{nullptr};
};

struct EdgeKernelHelperObjects : public HelperObjectsBase
{
  EdgeKernelHelperObjects(
    std::shared_ptr<stk::mesh::BulkData> bulk,
    stk::topology topo,
    int numDof,
    stk::mesh::Part* part)
    : HelperObjectsBase(bulk),
      linsys(new TestEdgeLinearSystem(realm, numDof, &eqSystem, topo))
  {
    eqSystem.linsys_ = linsys;
    edgeAlg.reset(
      new sierra::nalu::AssembleAMSEdgeKernelAlg(realm, part, &eqSystem));
  }

  virtual void execute() override
  {
    edgeAlg->execute();

    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  void print_lhs_and_rhs() const
  {
    HelperObjectsBase::print_lhs_and_rhs(linsys);
  }

  unit_test_utils::TestEdgeLinearSystem* linsys{nullptr};
  std::unique_ptr<sierra::nalu::AssembleAMSEdgeKernelAlg> edgeAlg;
};

struct NodeHelperObjects : public HelperObjectsBase
{
  NodeHelperObjects(
    std::shared_ptr<stk::mesh::BulkData> bulk,
    stk::topology topo,
    int numDof,
    stk::mesh::Part* part)
    : HelperObjectsBase(bulk),
      linsys(new TestEdgeLinearSystem(realm, numDof, &eqSystem, topo))
  {
    eqSystem.linsys_ = linsys;
    nodeAlg.reset(
      new sierra::nalu::AssembleNGPNodeSolverAlgorithm(realm, part, &eqSystem));
  }

  virtual void execute() override
  {
    nodeAlg->execute();

    Kokkos::deep_copy(linsys->hostNumSumIntoCalls_, linsys->numSumIntoCalls_);
    Kokkos::deep_copy(linsys->hostlhs_, linsys->lhs_);
    Kokkos::deep_copy(linsys->hostrhs_, linsys->rhs_);
  }

  void print_lhs_and_rhs() const
  {
    HelperObjectsBase::print_lhs_and_rhs(linsys);
  }

  unit_test_utils::TestEdgeLinearSystem* linsys{nullptr};
  std::unique_ptr<sierra::nalu::AssembleNGPNodeSolverAlgorithm> nodeAlg;
};

} // namespace unit_test_utils

#endif
