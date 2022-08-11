// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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
    ScalarFieldType*,
    const bool = false);

  virtual ~ScalarEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_{stk::mesh::InvalidOrdinal};
  unsigned scalarQ_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned dqdx_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_{stk::mesh::InvalidOrdinal};

  PecletFunction<AssembleEdgeSolverAlgorithm::DblType>* pecletFunction_{
    nullptr};

  std::string dofName_;
};

} // namespace nalu
} // namespace sierra

#endif /* SCALAREDGESOLVERALG_H */
