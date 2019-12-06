// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
  const DblType v2cMu_;
  const bool meshMotion_;

  // FIXME: What to do with alpha_kol in SST? This needs some thought...
  static constexpr double alpha_kol = 0.1;

  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned resAdeq_{stk::mesh::InvalidOrdinal};
  unsigned turbKineticEnergy_{stk::mesh::InvalidOrdinal};
  unsigned specDissipationRate_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgDudx_{stk::mesh::InvalidOrdinal};
  unsigned avgTkeRes_{stk::mesh::InvalidOrdinal};
  unsigned avgProd_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeq_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};
};

} // namespace nalu
} // namespace sierra

#endif
