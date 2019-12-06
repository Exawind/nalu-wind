// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TAMSAVGMDOTEDGEALG_H
#define TAMSAVGMDOTEDGEALG_H

#include "Algorithm.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class TAMSAvgMdotEdgeAlg : public Algorithm
{
public:
  using DblType = double;

  TAMSAvgMdotEdgeAlg(Realm&, stk::mesh::Part*);

  virtual ~TAMSAvgMdotEdgeAlg() = default;

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

#endif /* TAMSAVGMDOTEDGEALG_H */
