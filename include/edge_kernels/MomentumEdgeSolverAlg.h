/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMEDGESOLVERALG_H
#define MOMENTUMEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "PecletFunction.h"

namespace sierra {
namespace nalu {

class MomentumEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  MomentumEdgeSolverAlg(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

  virtual ~MomentumEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned velocity_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned dudx_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};
  unsigned viscosity_ {stk::mesh::InvalidOrdinal};

  PecletFunction<AssembleEdgeSolverAlgorithm::DblType>* pecletFunction_{nullptr};
};

}  // nalu
}  // sierra



#endif /* MOMENTUMEDGESOLVERALG_H */
