// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WALLDISTEDGESOLVERALG_H
#define WALLDISTEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class WallDistEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  WallDistEdgeSolverAlg(Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~WallDistEdgeSolverAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* WALLDISTEDGESOLVERALG_H */
