// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VOFADVECTIONEDGEALG_H
#define VOFADVECTIONEDGEALG_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "PecletFunction.h"
#include <stk_mesh/base/Types.hpp>

namespace sierra {
namespace nalu {

class VOFAdvectionEdgeAlg : public AssembleEdgeSolverAlgorithm
{
public:
  // TODO: refactor to use FieldManager
  VOFAdvectionEdgeAlg(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*,
    ScalarFieldType*,
    VectorFieldType*,
    const bool = false);

  virtual ~VOFAdvectionEdgeAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned scalarQ_{stk::mesh::InvalidOrdinal};
  unsigned dqdx_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* VOFADVECTIONEDGEALG_H */
