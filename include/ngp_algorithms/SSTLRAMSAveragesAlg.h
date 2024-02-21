// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SSTLRAMSAveragesAlg_h
#define SSTLRAMSAveragesAlg_h

#include "AMSAveragesAlg.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class SSTLRAMSAveragesAlg : public AMSAveragesAlg
{
public:
  using DblType = double;

  SSTLRAMSAveragesAlg(Realm& realm, stk::mesh::Part* part);

  virtual ~SSTLRAMSAveragesAlg() = default;

  virtual void execute() override;

private:
  const DblType betaStar_;
  const DblType CMdeg_;
  const DblType v2cMu_;
  const DblType aspectRatioSwitch_;
  const DblType alphaPow_;
  const DblType alphaScaPow_;
  const DblType coeffR_;
  const bool meshMotion_;

  unsigned velocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned dudx_{stk::mesh::InvalidOrdinal};
  unsigned resAdeq_{stk::mesh::InvalidOrdinal};
  unsigned turbKineticEnergy_{stk::mesh::InvalidOrdinal};
  unsigned specDissipationRate_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocityN_{stk::mesh::InvalidOrdinal};
  unsigned avgDudx_{stk::mesh::InvalidOrdinal};
  unsigned avgDudxN_{stk::mesh::InvalidOrdinal};
  unsigned avgTkeRes_{stk::mesh::InvalidOrdinal};
  unsigned avgTkeResN_{stk::mesh::InvalidOrdinal};
  unsigned avgProd_{stk::mesh::InvalidOrdinal};
  unsigned avgProdN_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeq_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeqN_{stk::mesh::InvalidOrdinal};
  unsigned fOneBlend_{stk::mesh::InvalidOrdinal};
  unsigned tvisc_{stk::mesh::InvalidOrdinal};
  unsigned visc_{stk::mesh::InvalidOrdinal};
  unsigned beta_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};
  unsigned wallDist_{stk::mesh::InvalidOrdinal};

  unsigned magPM_{stk::mesh::InvalidOrdinal};
  unsigned PMmax_{stk::mesh::InvalidOrdinal};
  unsigned PM1_{stk::mesh::InvalidOrdinal};
  unsigned PM2_{stk::mesh::InvalidOrdinal};
  unsigned PM3_{stk::mesh::InvalidOrdinal};

  // Proper definition of beta_kol in SSTLR-AMS doesn't work
  // near walls, so emprically tested floor is used currently
  static constexpr double beta_kol = 0.01;
};

} // namespace nalu
} // namespace sierra

#endif
