/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALAREDGESOLVERALG_H
#define SCALAREDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "PecletFunction.h"

namespace sierra {
namespace nalu {

class ScalarEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  ScalarEdgeSolverAlg(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*,
    ScalarFieldType*,
    VectorFieldType*,
    ScalarFieldType*);

  virtual ~ScalarEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned scalarQ_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned dqdx_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_ {stk::mesh::InvalidOrdinal};

  PecletFunction<AssembleEdgeSolverAlgorithm::DblType>* pecletFunction_{nullptr};

  std::string dofName_;
};

}  // nalu
}  // sierra


#endif /* SCALAREDGESOLVERALG_H */
