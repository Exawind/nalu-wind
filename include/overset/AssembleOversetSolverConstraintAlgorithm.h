/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleOversetSolverConstraintAlgorithm_h
#define AssembleOversetSolverConstraintAlgorithm_h

#include "overset/OversetConstraintBase.h"

namespace stk {
namespace mesh {
class Part;
class FieldBase;
}
}

namespace sierra{
namespace nalu{

class Realm;

class AssembleOversetSolverConstraintAlgorithm : public OversetConstraintBase
{
public:

  AssembleOversetSolverConstraintAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    stk::mesh::FieldBase *fieldQ);

  virtual ~AssembleOversetSolverConstraintAlgorithm() = default;

  virtual void execute();

private:
  AssembleOversetSolverConstraintAlgorithm() = delete;
  AssembleOversetSolverConstraintAlgorithm(
    const AssembleOversetSolverConstraintAlgorithm&) = delete;
};

} // namespace nalu
} // namespace Sierra

#endif
