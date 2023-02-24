// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMEDGESOLVERALG_H
#define MOMENTUMEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "PecletFunction.h"

namespace sierra {
namespace nalu {

class MomentumEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  MomentumEdgeSolverAlg(Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~MomentumEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned pecletFactor_{stk::mesh::InvalidOrdinal};
  unsigned maskNodeField_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMEDGESOLVERALG_H */
