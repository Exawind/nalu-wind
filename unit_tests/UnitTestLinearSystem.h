// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTLINEARSYSTEM_H
#define UNITTESTLINEARSYSTEM_H

#include "LinearSystem.h"
#include "EquationSystem.h"
#include "utils/CreateDeviceExpression.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace unit_test_utils {

template <
  typename InputLhsType,
  typename InputRhsType,
  typename LHSView,
  typename RHSView>
KOKKOS_FUNCTION void
assign(
  const InputLhsType& inputLhs,
  const InputRhsType& inputRhs,
  LHSView lhs,
  RHSView rhs)
{
  for (size_t i = 0; i < inputRhs.extent(0); ++i) {
    rhs(i) = inputRhs(i);
  }
  for (size_t i = 0; i < inputLhs.extent(0); ++i) {
    for (size_t j = 0; j < inputLhs.extent(1); ++j) {
      lhs(i, j) = inputLhs(i, j);
    }
  }
}

template <typename LHSView, typename RHSView>
KOKKOS_FUNCTION void
edgeSumInto(
  unsigned numEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  const sierra::nalu::SharedMemView<const double*, sierra::nalu::DeviceShmem>&
    rhs,
  const sierra::nalu::SharedMemView<const double**, sierra::nalu::DeviceShmem>&
    lhs,
  unsigned numDof,
  RHSView rhs_,
  LHSView lhs_)
{
  for (unsigned i = 0; i < numEntities; ++i) {
    auto ioff = (entities[i].local_offset() - 1) * numDof;
    for (unsigned d = 0; d < numDof; ++d)
      Kokkos::atomic_add(&rhs_(ioff + d), rhs(i * numDof + d));
  }

  for (unsigned i = 0; i < numEntities; ++i) {
    auto ioff = (entities[i].local_offset() - 1) * numDof;
    for (unsigned j = 0; j < numEntities; ++j) {
      auto joff = (entities[j].local_offset() - 1) * numDof;
      for (unsigned d = 0; d < numDof; ++d) {
        auto ii = i * numDof + d;
        auto jj = j * numDof + d;
        Kokkos::atomic_add(&lhs_(ioff + d, joff + d), lhs(ii, jj));
      }
    }
  }
}

using UnsignedView = Kokkos::View<unsigned*>;
using LHSView = Kokkos::View<double**>;
using RHSView = Kokkos::View<double*>;

class TestCoeffApplier : public sierra::nalu::CoeffApplier
{
public:
  KOKKOS_FUNCTION
  TestCoeffApplier(
    LHSView const& lhs,
    RHSView const& rhs,
    UnsignedView const& numSumIntoCalls,
    const bool isEdge = false,
    const unsigned nDof = 1,
    const unsigned numContributionsToAccept = 1)
    : numSumIntoCalls_(numSumIntoCalls),
      lhs_(lhs),
      rhs_(rhs),
      devicePointer_(nullptr),
      numContributionsToAccept_(numContributionsToAccept),
      isEdge_(isEdge),
      numDof_(nDof)
  {
  }

  KOKKOS_DEFAULTED_FUNCTION
  TestCoeffApplier(const TestCoeffApplier&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  ~TestCoeffApplier() = default;

  KOKKOS_FUNCTION
  void resetRows(
    unsigned /*numNodes*/,
    const stk::mesh::Entity* /*nodeList*/,
    const unsigned /*beginPos*/,
    const unsigned /*endPos*/,
    const double /*diag_value*/,
    const double /*rhs_residual*/)
  {
  }

  KOKKOS_FUNCTION
  void operator()(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /*localIds*/,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /*sortPermutation*/,
    const sierra::nalu::SharedMemView<const double*, sierra::nalu::DeviceShmem>&
      rhs,
    const sierra::nalu::
      SharedMemView<const double**, sierra::nalu::DeviceShmem>& lhs,
    const char* /*trace_tag*/)
  {
    if (isEdge_) {
      edgeSumInto(numEntities, entities, rhs, lhs, numDof_, rhs_, lhs_);
    } else {
      if (numSumIntoCalls_(0) < numContributionsToAccept_) {
        assign(lhs, rhs, lhs_, rhs_);
      }
    }
    Kokkos::atomic_add(&numSumIntoCalls_(0), 1u);
  }

  void free_device_pointer()
  {
    if (this != devicePointer_) {
      sierra::nalu::kokkos_free_on_device(devicePointer_);
      devicePointer_ = nullptr;
    }
  }

  sierra::nalu::CoeffApplier* device_pointer()
  {
    if (devicePointer_ != nullptr) {
      sierra::nalu::kokkos_free_on_device(devicePointer_);
      devicePointer_ = nullptr;
    }
    devicePointer_ = sierra::nalu::create_device_expression(*this);
    return devicePointer_;
  }

private:
  UnsignedView numSumIntoCalls_;
  LHSView lhs_;
  RHSView rhs_;
  TestCoeffApplier* devicePointer_;
  unsigned numContributionsToAccept_;
  bool isEdge_;
  unsigned numDof_;
};

class TestLinearSystem : public sierra::nalu::LinearSystem
{
public:
  TestLinearSystem(
    sierra::nalu::Realm& realm,
    const unsigned numDof,
    sierra::nalu::EquationSystem* eqSys,
    stk::topology topo,
    bool isEdge = false)
    : sierra::nalu::LinearSystem(realm, numDof, eqSys, nullptr),
      numDof_(numDof),
      isEdge_(isEdge)
  {
    unsigned rhsSize = numDof * topo.num_nodes();

    numSumIntoCalls_ = UnsignedView("numSumIntoCalls_", 1);

    rhs_ = RHSView("rhs_", rhsSize);
    lhs_ = LHSView("lhs_", rhsSize, rhsSize);

    hostNumSumIntoCalls_ = Kokkos::create_mirror_view(numSumIntoCalls_);
    hostrhs_ = Kokkos::create_mirror_view(rhs_);
    hostlhs_ = Kokkos::create_mirror_view(lhs_);
  }

  int getRowLID(stk::mesh::Entity node) const
  {
    return node.local_offset() - 1;
  }
  int getColLID(stk::mesh::Entity node) const
  {
    return node.local_offset() - 1;
  }

  virtual ~TestLinearSystem() {}

  // Graph/Matrix Construction
  virtual void buildNodeGraph(const stk::mesh::PartVector& /* parts */) {}
  virtual void buildFaceToNodeGraph(const stk::mesh::PartVector& /* parts */) {}
  virtual void buildEdgeToNodeGraph(const stk::mesh::PartVector& /* parts */) {}
  virtual void buildElemToNodeGraph(const stk::mesh::PartVector& /* parts */) {}
  virtual void
  buildReducedElemToNodeGraph(const stk::mesh::PartVector& /* parts */)
  {
  }
  virtual void
  buildFaceElemToNodeGraph(const stk::mesh::PartVector& /* parts */)
  {
  }
  virtual void
  buildNonConformalNodeGraph(const stk::mesh::PartVector& /* parts */)
  {
  }
  virtual void buildOversetNodeGraph(const stk::mesh::PartVector& /* parts */)
  {
  }
  virtual void finalizeLinearSystem() {}

  // Matrix Assembly
  virtual void zeroSystem() {}

  virtual void sumInto(
    unsigned /* numEntities */,
    const stk::mesh::Entity* /* entities */,
    const sierra::nalu::SharedMemView<const double*>& rhs,
    const sierra::nalu::SharedMemView<const double**>& lhs,
    const sierra::nalu::SharedMemView<int*>& /* localIds */,
    const sierra::nalu::SharedMemView<int*>& /* sortPermutation */,
    const char* /* trace_tag */
  )
  {
    if (numSumIntoCalls_(0) == 0) {
      assign(lhs, rhs, lhs_, rhs_);
    }
    Kokkos::atomic_add(&numSumIntoCalls_(0), 1u);
  }

  sierra::nalu::CoeffApplier* get_coeff_applier()
  {
    auto lhs = lhs_;
    auto rhs = rhs_;
    auto numSumIntoCalls = numSumIntoCalls_;
    auto isEdge = isEdge_;
    auto numDof = numDof_;

    auto newDeviceCoeffApplier =
      sierra::nalu::kokkos_malloc_on_device<TestCoeffApplier>(
        "deviceCoeffApplier");

    Kokkos::parallel_for(1, [=] KOKKOS_FUNCTION(const int&) {
      new (newDeviceCoeffApplier)
        TestCoeffApplier(lhs, rhs, numSumIntoCalls, isEdge, numDof);
    });

    return newDeviceCoeffApplier;
  }

  bool owns_coeff_applier() override { return false; }

  virtual void sumInto(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const sierra::nalu::SharedMemView<const double*, sierra::nalu::DeviceShmem>&
      rhs,
    const sierra::nalu::
      SharedMemView<const double**, sierra::nalu::DeviceShmem>& lhs,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /* localIds */,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /* sortPermutation */,
    const char* /* trace_tag */
  )
  {
    if (isEdge_) {
      edgeSumInto(numEntities, entities, rhs, lhs, numDof_, rhs_, lhs_);
    } else {
      if (numSumIntoCalls_(0) == 0) {
        assign(lhs, rhs, lhs_, rhs_);
      }
    }
    Kokkos::atomic_add(&numSumIntoCalls_(0), 1u);
  }

  virtual void sumInto(
    const std::vector<stk::mesh::Entity>& /* sym_meshobj */,
    std::vector<int>& /* scratchIds */,
    std::vector<double>& /* scratchVals */,
    const std::vector<double>& rhs,
    const std::vector<double>& lhs,
    const char* /* trace_tag */ = 0)
  {
    if (numSumIntoCalls_(0) == 0) {
      for (size_t i = 0; i < rhs.size(); ++i) {
        rhs_(i) = rhs[i];
      }
      const size_t numRows = rhs.size();
      ThrowAssert(numRows * numRows == lhs.size());
      for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numRows; ++j) {
          lhs_(i, j) = lhs[numRows * i + j];
        }
      }
    }
    Kokkos::atomic_add(&numSumIntoCalls_(0), 1u);
  }

  virtual void applyDirichletBCs(
    stk::mesh::FieldBase* /* solutionField */,
    stk::mesh::FieldBase* /* bcValuesField */,
    const stk::mesh::PartVector& /* parts */,
    const unsigned /* beginPos */,
    const unsigned /* endPos */)
  {
  }

  // Solve
  virtual int solve(stk::mesh::FieldBase* /* linearSolutionField */)
  {
    return -1;
  }
  virtual void loadComplete() {}

  virtual void
  writeToFile(const char* /* filename */, bool /* useOwned */ = true)
  {
  }
  virtual void
  writeSolutionToFile(const char* /* filename */, bool /* useOwned */ = true)
  {
  }

  virtual void resetRows(
    const std::vector<stk::mesh::Entity>& /* nodeList */,
    const unsigned /* beginPos */,
    const unsigned /* endPos */,
    const double,
    const double)
  {
  }

  virtual void resetRows(
    unsigned /*numNodes*/,
    const stk::mesh::Entity* /* nodeList */,
    const unsigned /* beginPos */,
    const unsigned /* endPos */,
    const double,
    const double)
  {
  }

  UnsignedView numSumIntoCalls_;
  UnsignedView::HostMirror hostNumSumIntoCalls_;
  LHSView lhs_;
  LHSView::HostMirror hostlhs_;
  RHSView rhs_;
  RHSView::HostMirror hostrhs_;
  unsigned numDof_;
  bool isEdge_;

protected:
  virtual void beginLinearSystemConstruction() {}
  virtual void checkError(const int /* err_code */, const char* /* msg */) {}
};

class TestEdgeLinearSystem : public TestLinearSystem
{
public:
  TestEdgeLinearSystem(
    sierra::nalu::Realm& realm,
    const unsigned numDof,
    sierra::nalu::EquationSystem* eqSys,
    stk::topology topo)
    : TestLinearSystem(realm, numDof, eqSys, topo, true /*isEdge*/),
      numDof_(numDof)
  {
  }

  using TestLinearSystem::sumInto;
  virtual void sumInto(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const sierra::nalu::SharedMemView<const double*, sierra::nalu::DeviceShmem>&
      rhs,
    const sierra::nalu::
      SharedMemView<const double**, sierra::nalu::DeviceShmem>& lhs,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /* localIds */,
    const sierra::nalu::
      SharedMemView<int*, sierra::nalu::DeviceShmem>& /* sortPermutation */,
    const char* /* trace_tag */)
  {
    edgeSumInto(numEntities, entities, rhs, lhs, numDof_, rhs_, lhs_);
    Kokkos::atomic_add(&numSumIntoCalls_(0), 1u);
  }

  unsigned numDof_;
};

} // namespace unit_test_utils

#endif /* UNITTESTLINEARSYSTEM_H */
