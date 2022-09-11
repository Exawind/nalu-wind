// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SolverAlgorithm_h
#define SolverAlgorithm_h

#include <Algorithm.h>
#include <KokkosInterface.h>
#include <LinearSystem.h>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <vector>

namespace sierra {
namespace nalu {

class CoeffApplier;
class EquationSystem;
class Realm;

struct NGPApplyCoeff
{
  NGPApplyCoeff(EquationSystem*);

  KOKKOS_DEFAULTED_FUNCTION
  NGPApplyCoeff() = delete;

  KOKKOS_DEFAULTED_FUNCTION
  ~NGPApplyCoeff() = default;

  void free_coeff_applier();

  KOKKOS_FUNCTION
  void operator()(
    unsigned numMeshobjs,
    const stk::mesh::NgpMesh::ConnectedNodes& symMeshobjs,
    const SharedMemView<int*, DeviceShmem>& scratchIds,
    const SharedMemView<int*, DeviceShmem>& sortPermutation,
    SharedMemView<double*, DeviceShmem>& rhs,
    SharedMemView<double**, DeviceShmem>& lhs,
    const char* trace_tag) const;

  KOKKOS_FUNCTION
  void extract_diagonal(
    const unsigned nEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    SharedMemView<double**, DeviceShmem>& lhs) const;

  KOKKOS_FUNCTION
  void reset_overset_rows(
    const unsigned nEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    SharedMemView<double*, DeviceShmem>& rhs,
    SharedMemView<double**, DeviceShmem>& lhs) const;

  const stk::mesh::NgpMesh ngpMesh_;
  mutable stk::mesh::NgpField<double> diagField_;
  stk::mesh::NgpField<int> iblankField_;

  CoeffApplier* deviceSumInto_;

  const unsigned nDim_{3};
  const bool hasOverset_{false};
  const bool extractDiagonal_{false};
  const bool resetOversetRows_{true};
  const bool linSysOwnsCoeffApplier;
};

class SolverAlgorithm : public Algorithm
{
public:
  SolverAlgorithm(
    Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem);
  virtual ~SolverAlgorithm() {}

  virtual void execute() = 0;
  virtual void initialize_connectivity() = 0;

protected:
  NGPApplyCoeff coeff_applier() { return NGPApplyCoeff(eqSystem_); }

  // Need to find out whether this ever gets called inside a modification cycle.
  void apply_coeff(
    const std::vector<stk::mesh::Entity>& sym_meshobj,
    std::vector<int>& scratchIds,
    std::vector<double>& scratchVals,
    std::vector<double>& rhs,
    std::vector<double>& lhs,
    const char* trace_tag = 0);

  EquationSystem* eqSystem_;
};

} // namespace nalu
} // namespace sierra

#endif
