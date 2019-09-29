/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SSTTAMSAveragesAlg_h
#define SSTTAMSAveragesAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class SSTTAMSAveragesAlg : public Algorithm
{
public:
  using DblType = double;

  SSTTAMSAveragesAlg(Realm& realm, stk::mesh::Part* part);

  virtual ~SSTTAMSAveragesAlg() = default;

  virtual void execute() override;

private:
  const DblType betaStar_;
  const DblType CMdeg_;
  const bool meshMotion_;

  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned resAdeq_{stk::mesh::InvalidOrdinal};
  unsigned turbKineticEnergy_{stk::mesh::InvalidOrdinal};
  unsigned specDissipationRate_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityNP1_{stk::mesh::InvalidOrdinal};
  unsigned avgDudxNP1_{stk::mesh::InvalidOrdinal};
  unsigned avgTkeResNP1_{stk::mesh::InvalidOrdinal};
  unsigned avgProdNP1_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqNP1_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};

  unsigned avgVelocityN_{stk::mesh::InvalidOrdinal};
  unsigned avgDudxN_{stk::mesh::InvalidOrdinal};
  unsigned avgTkeResN_{stk::mesh::InvalidOrdinal};
  unsigned avgProdN_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqN_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif
