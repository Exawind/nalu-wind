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

#include <stk_mesh/base/Entity.hpp>

#include <vector>

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
  const std::vector<double> & rhs,
  const std::vector<double> & lhs, const char *trace_tag)
{
  eqSystem_->linsys_->sumInto(sym_meshobj, scratchIds, scratchVals, rhs, lhs, trace_tag);
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
}

} // namespace nalu
} // namespace Sierra
