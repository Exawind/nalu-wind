// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SDRLOWREWALLFUNCALG_H
#define SDRLOWREWALLFUNCALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

/** Specific Dissipation Rate low-Reynolds number wall boundary value
 *
 *. \sa SDRWallFuncAlgDriver
 */
template <typename BcAlgTraits>
class SDRLowReWallAlg : public Algorithm
{
public:
  SDRLowReWallAlg(Realm&, stk::mesh::Part*, const bool);

  virtual ~SDRLowReWallAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests faceData_;
  ElemDataRequests elemData_;

  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned wallArea_{stk::mesh::InvalidOrdinal};
  unsigned sdrbc_{stk::mesh::InvalidOrdinal};

  const DoubleType betaOne_;
  const DoubleType wallFactor_;
  const bool useShifted_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* SDRLOWREWALLFUNCALG_H */
