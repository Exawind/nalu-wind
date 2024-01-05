// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SolverAlgorithm.h>
#include <Algorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <KokkosInterface.h>

#include "ngp_utils/NgpFieldUtils.h"

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

#include <vector>

namespace {

inline void
fix_overset_rows(
  const stk::mesh::MetaData& meta,
  const size_t nDim,
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<double>& rhs,
  std::vector<double>& lhs)
{
  using ScalarIntFieldType = sierra::nalu::ScalarIntFieldType;
  const size_t nobj = entities.size();
  const size_t numRows = nobj * nDim;

  ScalarIntFieldType* iblank =
    meta.get_field<ScalarIntFieldType>(stk::topology::NODE_RANK, "iblank");

  for (size_t in = 0; in < nobj; in++) {
    const int* ibl = stk::mesh::field_data(*iblank, entities[in]);
    double mask = std::max(0.0, static_cast<double>(ibl[0]));
    size_t ix = in * nDim;

    for (size_t d = 0; d < nDim; d++) {
      size_t ir = ix + d;
      rhs[ir] *= mask;
      for (size_t c = 0; c < numRows; c++)
        lhs[ir * numRows + c] *= mask;
    }
  }
}
} // namespace

namespace sierra {
namespace nalu {

NGPApplyCoeff::NGPApplyCoeff(EquationSystem* eqSystem)
  : ngpMesh_(eqSystem->realm_.ngp_mesh()),
    deviceSumInto_(eqSystem->linsys_->get_coeff_applier()),
    nDim_(eqSystem->linsys_->numDof()),
    hasOverset_(eqSystem->realm_.hasOverset_),
    extractDiagonal_(eqSystem->extractDiagonal_),
    resetOversetRows_(eqSystem->resetOversetRows_),
    linSysOwnsCoeffApplier(eqSystem->linsys_->owns_coeff_applier())
{
  if (extractDiagonal_) {
    diagField_ = nalu_ngp::get_ngp_field(
      eqSystem->realm_.mesh_info(), eqSystem->get_diagonal_field()->name());
  }

  if (hasOverset_) {
    iblankField_ =
      nalu_ngp::get_ngp_field<int>(eqSystem->realm_.mesh_info(), "iblank");
  }
}

void
NGPApplyCoeff::free_coeff_applier()
{
  if (deviceSumInto_ != nullptr && !linSysOwnsCoeffApplier) {
    kokkos_free_on_device(deviceSumInto_);
    deviceSumInto_ = nullptr;
  }
}

KOKKOS_FUNCTION
void
NGPApplyCoeff::extract_diagonal(
  const unsigned int nEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  SharedMemView<double**, DeviceShmem>& lhs) const
{
  constexpr bool forceAtomic = std::is_same<
    sierra::nalu::DeviceSpace, Kokkos::DefaultExecutionSpace>::value;

  for (unsigned i = 0u; i < nEntities; ++i) {
    auto ix = i * nDim_;
    if (forceAtomic)
      Kokkos::atomic_add(
        &diagField_.get(ngpMesh_, entities[i], 0), lhs(ix, ix));
    else
      diagField_.get(ngpMesh_, entities[i], 0) += lhs(ix, ix);
  }
}

KOKKOS_FUNCTION
void
NGPApplyCoeff::reset_overset_rows(
  const unsigned int nEntities,
  const stk::mesh::NgpMesh::ConnectedNodes& entities,
  SharedMemView<double*, DeviceShmem>& rhs,
  SharedMemView<double**, DeviceShmem>& lhs) const
{
  const unsigned numRows = nEntities * nDim_;

  for (unsigned in = 0u; in < nEntities; ++in) {
    const int ibl = iblankField_.get(ngpMesh_, entities[in], 0);
    const double mask = stk::math::max(0.0, static_cast<double>(ibl));
    const unsigned ix = in * nDim_;

    for (unsigned d = 0; d < nDim_; ++d) {
      const unsigned ir = ix + d;

      rhs(ir) *= mask;
      for (unsigned ic = 0; ic < numRows; ++ic)
        lhs(ir, ic) *= mask;
    }
  }
}

KOKKOS_FUNCTION
void
NGPApplyCoeff::operator()(
  unsigned numMeshobjs,
  const stk::mesh::NgpMesh::ConnectedNodes& symMeshobjs,
  const SharedMemView<int*, DeviceShmem>& scratchIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  SharedMemView<double*, DeviceShmem>& rhs,
  SharedMemView<double**, DeviceShmem>& lhs,
  const char* trace_tag) const
{
  if (extractDiagonal_)
    extract_diagonal(numMeshobjs, symMeshobjs, lhs);

  if (hasOverset_ && resetOversetRows_)
    reset_overset_rows(numMeshobjs, symMeshobjs, rhs, lhs);

  (*deviceSumInto_)(
    numMeshobjs, symMeshobjs, scratchIds, sortPermutation, rhs, lhs, trace_tag);
}

SolverAlgorithm::SolverAlgorithm(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : Algorithm(realm, part), eqSystem_(eqSystem)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- apply_coeff -----------------------------------------------------
//--------------------------------------------------------------------------
void
SolverAlgorithm::apply_coeff(
  const std::vector<stk::mesh::Entity>& sym_meshobj,
  std::vector<int>& scratchIds,
  std::vector<double>& scratchVals,
  std::vector<double>& rhs,
  std::vector<double>& lhs,
  const char* trace_tag)
{
  if (realm_.hasOverset_)
    fix_overset_rows(
      realm_.meta_data(), eqSystem_->linsys_->numDof(), sym_meshobj, rhs, lhs);

  eqSystem_->linsys_->sumInto(
    sym_meshobj, scratchIds, scratchVals, rhs, lhs, trace_tag);

  if (eqSystem_->extractDiagonal_)
    eqSystem_->save_diagonal_term(sym_meshobj, scratchIds, lhs);
}

} // namespace nalu
} // namespace sierra
