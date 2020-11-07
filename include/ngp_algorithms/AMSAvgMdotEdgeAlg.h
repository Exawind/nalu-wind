// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef AMSAVGMDOTEDGEALG_H
#define AMSAVGMDOTEDGEALG_H

#include "Algorithm.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class AMSAvgMdotEdgeAlg : public Algorithm
{
public:
  using DblType = double;

  AMSAvgMdotEdgeAlg(Realm&, stk::mesh::Part*);

  virtual ~AMSAvgMdotEdgeAlg() = default;

  void execute() override;

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityRTM_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned avgMassFlowRate_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* AMSAVGMDOTEDGEALG_H */
