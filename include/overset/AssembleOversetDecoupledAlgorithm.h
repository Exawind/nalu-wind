/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ASSEMBLEOVERSETDECOUPLEDALGORITHM_H
#define ASSEMBLEOVERSETDECOUPLEDALGORITHM_H

#include "overset/OversetConstraintBase.h"

namespace stk {
namespace mesh {
class Part;
class FieldBase;
}
}

namespace sierra {
namespace nalu {

class Realm;

class AssembleOversetDecoupledAlgorithm : public OversetConstraintBase
{
public:
  AssembleOversetDecoupledAlgorithm(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*,
    stk::mesh::FieldBase*);

  AssembleOversetDecoupledAlgorithm() = delete;
  AssembleOversetDecoupledAlgorithm(const AssembleOversetDecoupledAlgorithm&) = delete;
  virtual ~AssembleOversetDecoupledAlgorithm() = default;

  virtual void execute() override;
};

}  // nalu
}  // sierra


#endif /* ASSEMBLEOVERSETDECOUPLEDALGORITHM_H */
