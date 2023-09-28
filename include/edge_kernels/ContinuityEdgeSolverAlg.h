// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONTINUITYEDGESOLVERALG_H
#define CONTINUITYEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class ContinuityEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  ContinuityEdgeSolverAlg(Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~ContinuityEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned pressure_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned Gpdx_{stk::mesh::InvalidOrdinal};
  unsigned source_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned edgeFaceVelMag_{stk::mesh::InvalidOrdinal};
  unsigned Udiag_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* CONTINUITYEDGESOLVERALG_H */
