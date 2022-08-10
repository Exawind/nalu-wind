// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTEDGEALG_H
#define MDOTEDGEALG_H

#include "Algorithm.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class MdotEdgeAlg : public Algorithm
{
public:
  using DblType = double;

  MdotEdgeAlg(Realm&, stk::mesh::Part*);

  virtual ~MdotEdgeAlg() = default;

  void execute() override;

private:
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned pressure_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned Gpdx_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned edgeFaceVelMag_{stk::mesh::InvalidOrdinal};
  unsigned Udiag_{stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif /* MDOTEDGEALG_H */
