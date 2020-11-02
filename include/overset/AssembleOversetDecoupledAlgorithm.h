// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
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
