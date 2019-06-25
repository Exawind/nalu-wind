/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTLINEARSYSTEM_H
#define UNITTESTLINEARSYSTEM_H

#include "LinearSystem.h"
#include "EquationSystem.h"
#include "utils/CreateDeviceExpression.h"

namespace unit_test_utils {

template<typename InputLhsType, typename InputRhsType,
         typename LHSView, typename RHSView>
KOKKOS_FUNCTION
void assign(const InputLhsType& inputLhs, const InputRhsType& inputRhs,
            LHSView lhs, RHSView rhs)
{
    for(size_t i=0; i<inputRhs.extent(0); ++i) {
      rhs(i) = inputRhs(i);
    }
    for(size_t i=0; i<inputLhs.extent(0); ++i) {
      for(size_t j=0; j<inputLhs.extent(1); ++j) {
        lhs(i,j) = inputLhs(i,j);
      }
    }
}

using LHSView = Kokkos::View<double**>;
using RHSView = Kokkos::View<double*>;

class TestCoeffApplier : public sierra::nalu::CoeffApplier
{
public:
  KOKKOS_FUNCTION
  TestCoeffApplier(LHSView& lhs, RHSView& rhs, unsigned numContributionsToAccept = 1)
  : lhs_(lhs), rhs_(rhs), devicePointer_(nullptr),
    numContributionsToAccept_(numContributionsToAccept), numContributions_(0)
  {}

  KOKKOS_FUNCTION
  TestCoeffApplier(const TestCoeffApplier&) = default;

  KOKKOS_FUNCTION
  ~TestCoeffApplier()
  {
  }

  KOKKOS_FUNCTION
  void operator()(unsigned /*numEntities*/,
                  const ngp::Mesh::ConnectedNodes& /*entities*/,
                  const sierra::nalu::SharedMemView<int*, sierra::nalu::DeviceShmem> & /*localIds*/,
                  const sierra::nalu::SharedMemView<int*, sierra::nalu::DeviceShmem> & /*sortPermutation*/,
                  const sierra::nalu::SharedMemView<const double*, sierra::nalu::DeviceShmem> & rhs,
                  const sierra::nalu::SharedMemView<const double**, sierra::nalu::DeviceShmem> & lhs,
                  const char * /*trace_tag*/)
  {
    if (numContributions_ < numContributionsToAccept_) {
      assign(lhs, rhs, lhs_, rhs_);
      ++numContributions_;
    }
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
  LHSView lhs_;
  RHSView rhs_;
  TestCoeffApplier* devicePointer_;
  unsigned numContributionsToAccept_;
  unsigned numContributions_;
};

class TestLinearSystem : public sierra::nalu::LinearSystem
{
public:
 TestLinearSystem( sierra::nalu::Realm &realm, const unsigned numDof,
                   sierra::nalu::EquationSystem *eqSys, stk::topology topo)
   : sierra::nalu::LinearSystem(realm, numDof, eqSys, nullptr), numSumIntoCalls_(0)
  {
    unsigned rhsSize = numDof * topo.num_nodes();

    rhs_ = RHSView("rhs_",rhsSize);
    lhs_ = LHSView("lhs_",rhsSize, rhsSize);

    hostrhs_ = Kokkos::create_mirror_view(rhs_);
    hostlhs_ = Kokkos::create_mirror_view(lhs_);
  }

  virtual ~TestLinearSystem() {}

  // Graph/Matrix Construction
  virtual void buildNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildFaceToNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildEdgeToNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildElemToNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildReducedElemToNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildFaceElemToNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildNonConformalNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void buildOversetNodeGraph(const stk::mesh::PartVector &  /* parts */) {}
  virtual void finalizeLinearSystem() {}

  // Matrix Assembly
  virtual void zeroSystem() {}

  virtual void sumInto(
      unsigned  /* numEntities */,
      const stk::mesh::Entity*  /* entities */,
      const sierra::nalu::SharedMemView<const double*> & rhs,
      const sierra::nalu::SharedMemView<const double**> & lhs,
      const sierra::nalu::SharedMemView<int*> &  /* localIds */,
      const sierra::nalu::SharedMemView<int*> &  /* sortPermutation */,
      const char *  /* trace_tag */
      )
  {
    if (numSumIntoCalls_ == 0) {
      assign(lhs, rhs, lhs_, rhs_);
    }
    Kokkos::atomic_add(&numSumIntoCalls_, 1u);
  }

  sierra::nalu::CoeffApplier* get_coeff_applier()
  {
    return new TestCoeffApplier(lhs_, rhs_);
  }

  virtual void sumInto(
    unsigned  /* numEntities */,
    const ngp::Mesh::ConnectedNodes&  /* entities */,
    const sierra::nalu::SharedMemView<const double*,sierra::nalu::DeviceShmem> & rhs,
    const sierra::nalu::SharedMemView<const double**,sierra::nalu::DeviceShmem> & lhs,
    const sierra::nalu::SharedMemView<int*,sierra::nalu::DeviceShmem> &  /* localIds */,
    const sierra::nalu::SharedMemView<int*,sierra::nalu::DeviceShmem> &  /* sortPermutation */,
    const char *  /* trace_tag */
  )
  {
    if (numSumIntoCalls_ == 0) {
      assign(lhs, rhs, lhs_, rhs_);
    }
    Kokkos::atomic_add(&numSumIntoCalls_, 1u);
  }

  virtual void sumInto(
    const std::vector<stk::mesh::Entity> &  /* sym_meshobj */,
    std::vector<int> & /* scratchIds */,
    std::vector<double> & /* scratchVals */,
    const std::vector<double> & rhs,
    const std::vector<double> & lhs,
    const char * /* trace_tag */=0
    )
  {
    if (numSumIntoCalls_ == 0) {
      for (size_t i=0; i<rhs.size(); ++i) {
        rhs_(i) = rhs[i];
      }
      const size_t numRows = rhs.size();
      ThrowAssert(numRows*numRows == lhs.size());
      for (size_t i=0; i<numRows; ++i) {
        for (size_t j=0; j<numRows; ++j) {
          lhs_(i,j) = lhs[numRows*i+j];
        }
      }
    }
    numSumIntoCalls_++;
  }

  virtual void applyDirichletBCs(
    stk::mesh::FieldBase *  /* solutionField */,
    stk::mesh::FieldBase *  /* bcValuesField */,
    const stk::mesh::PartVector &  /* parts */,
    const unsigned  /* beginPos */,
    const unsigned  /* endPos */)
  {}

  virtual void prepareConstraints(
    const unsigned  /* beginPos */,
    const unsigned  /* endPos */)
  {}

  // Solve
  virtual int solve(stk::mesh::FieldBase *  /* linearSolutionField */) { return -1; }
  virtual void loadComplete() {}

  virtual void writeToFile(const char *  /* filename */, bool  /* useOwned */=true) {}
  virtual void writeSolutionToFile(const char *  /* filename */, bool  /* useOwned */=true) {}

  virtual void resetRows(
    const std::vector<stk::mesh::Entity>&  /* nodeList */,
    const unsigned  /* beginPos */,
    const unsigned  /* endPos */,
    const double,
    const double) {}

  unsigned numSumIntoCalls_;
  LHSView lhs_;
  LHSView::HostMirror hostlhs_;
  RHSView rhs_;
  RHSView::HostMirror hostrhs_;

protected:
  virtual void beginLinearSystemConstruction() {}
  virtual void checkError(
    const int  /* err_code */,
    const char *  /* msg */) {}
};

class TestEdgeLinearSystem : public TestLinearSystem
{
public:
  TestEdgeLinearSystem(
    sierra::nalu::Realm& realm,
    const unsigned numDof,
    sierra::nalu::EquationSystem* eqSys,
    stk::topology topo
  ) : TestLinearSystem(realm, numDof, eqSys, topo)
  {}

  using TestLinearSystem::sumInto;
  virtual void sumInto(
    unsigned numEntities,
    const ngp::Mesh::ConnectedNodes&  entities,
    const sierra::nalu::SharedMemView<const double*,sierra::nalu::DeviceShmem> & rhs,
    const sierra::nalu::SharedMemView<const double**,sierra::nalu::DeviceShmem> & lhs,
    const sierra::nalu::SharedMemView<int*,sierra::nalu::DeviceShmem> &  /* localIds */,
    const sierra::nalu::SharedMemView<int*,sierra::nalu::DeviceShmem> &  /* sortPermutation */,
    const char *  /* trace_tag */)
  {
    for (unsigned i=0; i < numEntities; ++i) {
      auto ioff = (entities[i].local_offset() - 1) * numDof();
      for (unsigned d=0; d < numDof(); ++d)
        Kokkos::atomic_add(&rhs_(ioff + d), rhs(i * numDof() + d));
    }

    for (unsigned i=0; i < numEntities; ++i) {
      auto ioff = (entities[i].local_offset() - 1) * numDof();
      for (unsigned j=0; j < numEntities; ++j) {
        auto joff = (entities[j].local_offset() - 1) * numDof();
        for (unsigned d=0; d < numDof(); ++d) {
          auto ii = i * numDof() + d;
          auto jj = j * numDof() + d;
          Kokkos::atomic_add(&lhs_(ioff + d, joff + d), lhs(ii, jj));
        }
      }
    }
    Kokkos::atomic_add(&numSumIntoCalls_, 1u);
  }

};

} // namespace unit_test_utils

#endif /* UNITTESTLINEARSYSTEM_H */

