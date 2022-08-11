// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleOversetSolverConstraintAlgorithm_h
#define AssembleOversetSolverConstraintAlgorithm_h

#include "overset/OversetConstraintBase.h"

namespace stk {
namespace mesh {
class Part;
class FieldBase;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;

class AssembleOversetSolverConstraintAlgorithm : public OversetConstraintBase
{
public:
  AssembleOversetSolverConstraintAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    stk::mesh::FieldBase* fieldQ);

  virtual ~AssembleOversetSolverConstraintAlgorithm() = default;

  virtual void execute();

private:
  AssembleOversetSolverConstraintAlgorithm() = delete;
  AssembleOversetSolverConstraintAlgorithm(
    const AssembleOversetSolverConstraintAlgorithm&) = delete;
};

} // namespace nalu
} // namespace sierra

#endif
