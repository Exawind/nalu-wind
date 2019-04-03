/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <SolverAlgorithm.h>
#include <Algorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <KokkosInterface.h>

#include <stk_mesh/base/Entity.hpp>

#include <vector>

namespace {

inline
void fix_overset_rows(
  const stk::mesh::MetaData& meta,
  const size_t nDim,
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<double>& rhs,
  std::vector<double>& lhs)
{
  using ScalarIntFieldType = sierra::nalu::ScalarIntFieldType;
  const size_t nobj = entities.size();
  const size_t numRows = nobj * nDim;

  ScalarIntFieldType* iblank = meta.get_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "iblank");

  for (size_t in=0; in < nobj; in++) {
    const int* ibl = stk::mesh::field_data(*iblank, entities[in]);
    double mask = std::max(0.0, static_cast<double>(ibl[0]));
    size_t ix = in * nDim;

    for (size_t d=0; d < nDim; d++) {
      size_t ir = ix + d;
      rhs[ir] *= mask;
      for (size_t c=0; c < numRows; c++)
        lhs[ir * numRows + c] *= mask;
    }
  }
}

inline
void fix_overset_rows(
  const stk::mesh::MetaData& meta,
  const size_t nDim,
  const size_t nEntities,
  const stk::mesh::Entity* entities,
  const sierra::nalu::SharedMemView<double*>& rhs,
  const sierra::nalu::SharedMemView<double**>& lhs)
{
  using ScalarIntFieldType = sierra::nalu::ScalarIntFieldType;
  const size_t numRows = nEntities * nDim;

  ScalarIntFieldType* iblank = meta.get_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "iblank");

  for (size_t in=0; in < nEntities; in++) {
    const int* ibl = stk::mesh::field_data(*iblank, entities[in]);
    double mask = std::max(0.0, static_cast<double>(ibl[0]));
    size_t ix = in * nDim;

    for (size_t d=0; d < nDim; d++) {
      size_t ir = ix + d;

      rhs(ir) *= mask;
      for (size_t c=0; c < numRows; c++)
        lhs(ir, c) *= mask;
    }
  }
}
}

namespace sierra{
namespace nalu{

class Realm;
class EquationSystem;

//==========================================================================
// Class Definition
//==========================================================================
// SolverAlgorithm - base class for algorithm with expectations of solver
//                   contributions
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SolverAlgorithm::SolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : Algorithm(realm, part),
    eqSystem_(eqSystem)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- apply_coeff -----------------------------------------------------
//--------------------------------------------------------------------------
void
SolverAlgorithm::apply_coeff(
  const std::vector<stk::mesh::Entity> & sym_meshobj,
  std::vector<int> &scratchIds,
  std::vector<double> &scratchVals,
  std::vector<double> & rhs,
  std::vector<double> & lhs, const char *trace_tag)
{
  if (realm_.hasOverset_)
    fix_overset_rows(
      realm_.meta_data(), eqSystem_->linsys_->numDof(), sym_meshobj, rhs, lhs);

  eqSystem_->linsys_->sumInto(
    sym_meshobj, scratchIds, scratchVals, rhs, lhs, trace_tag);

  if (eqSystem_->extractDiagonal_)
    eqSystem_->save_diagonal_term(sym_meshobj, scratchIds, lhs);
}

void
SolverAlgorithm::apply_coeff(
  unsigned numMeshobjs,
  const stk::mesh::Entity* symMeshobjs,
  const SharedMemView<int*> & scratchIds,
  const SharedMemView<int*> & sortPermutation,
  SharedMemView<double*> & rhs,
  SharedMemView<double**> & lhs,
  const char *trace_tag)
{
  if (realm_.hasOverset_)
    fix_overset_rows(
      realm_.meta_data(), eqSystem_->linsys_->numDof(), numMeshobjs,
      symMeshobjs, rhs, lhs);

  eqSystem_->linsys_->sumInto(
    numMeshobjs, symMeshobjs, rhs, lhs, scratchIds, sortPermutation, trace_tag);

  if (eqSystem_->extractDiagonal_)
    eqSystem_->save_diagonal_term(numMeshobjs, symMeshobjs, lhs);
}

void
SolverAlgorithm::apply_coeff(
  unsigned numMeshobjs,
  const ngp::Mesh::ConnectedNodes& symMeshobjs,
  const SharedMemView<int*> & scratchIds,
  const SharedMemView<int*> & sortPermutation,
  const SharedMemView<const double*> & rhs,
  const SharedMemView<const double**> & lhs,
  const char *trace_tag)
{
  eqSystem_->linsys_->sumInto(numMeshobjs, symMeshobjs, rhs, lhs, scratchIds, sortPermutation, trace_tag);

  if (eqSystem_->extractDiagonal_)
    eqSystem_->save_diagonal_term(numMeshobjs, symMeshobjs, lhs);
}

} // namespace nalu
} // namespace Sierra
