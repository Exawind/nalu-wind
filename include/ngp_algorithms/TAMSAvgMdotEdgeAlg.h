/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
