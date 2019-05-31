/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CONTINUITYEDGESOLVERALG_H
#define CONTINUITYEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class ContinuityEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  ContinuityEdgeSolverAlg(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

  virtual ~ContinuityEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned pressure_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned Gpdx_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned Udiag_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* CONTINUITYEDGESOLVERALG_H */
