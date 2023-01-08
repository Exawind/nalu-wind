// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ZALESAKDISKMASSFLOWRATEALG_H
#define ZALESAKDISKMASSFLOWRATEALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class ZalesakDiskMassFlowRateEdgeAlg : public AssembleEdgeSolverAlgorithm
{
public:
  // TODO: refactor to use FieldManager
  ZalesakDiskMassFlowRateEdgeAlg(
    Realm&, stk::mesh::Part*, EquationSystem*, const bool = false);

  virtual ~ZalesakDiskMassFlowRateEdgeAlg() = default;

  virtual void execute();

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* ZALESAKDISKMASSFLOWRATEALG_H */
