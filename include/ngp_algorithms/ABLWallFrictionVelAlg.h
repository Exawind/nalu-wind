// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ABLWALLFRICTIONVELALG_H
#define ABLWALLFRICTIONVELALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "ngp_algorithms/WallFricVelAlgDriver.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

/** Compute the wall friction velocity at integration points for the wall
 *  boundary of a given topology.
 *
 *  In addition to computing the friction velocity (utau) at the integration
 *  points, it also computes a partial sum (utau * area) and (area) at the
 *  integration points that is used to compute the area-weighted average utau
 *  over the ABL wall by WallFricVelAlgDriver.
 *
 *  \sa WallFricVelAlgDriver, BdyLayerStatistics
 */
template <typename BcAlgTraits>
class ABLWallFrictionVelAlg : public Algorithm
{
public:
  using DblType = double;

  ABLWallFrictionVelAlg(
    Realm&,
    stk::mesh::Part*,
    WallFricVelAlgDriver&,
    const bool,
    const double,
    const double,
    const double,
    const double);

  virtual ~ABLWallFrictionVelAlg() = default;

  virtual void execute() override;

private:
  WallFricVelAlgDriver& algDriver_;

  ElemDataRequests faceData_;

  unsigned velocityNp1_     {stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_      {stk::mesh::InvalidOrdinal};
  unsigned density_         {stk::mesh::InvalidOrdinal};
  unsigned bcHeatFlux_      {stk::mesh::InvalidOrdinal};
  unsigned specificHeat_    {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_  {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_     {stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_    {stk::mesh::InvalidOrdinal};

  //! Acceleration due to gravity (m/s^2)
  const DoubleType gravity_;

  //! Roughness height (m)
  const DoubleType z0_;

  //! Reference temperature (K)
  const DoubleType Tref_;

  //! von Karman constant
  const DoubleType kappa_{0.41};
  const DoubleType beta_m_{5.0};
  const DoubleType beta_h_{5.0};
  const DoubleType gamma_m_{16.0};
  const DoubleType gamma_h_{16.0};

  bool useShifted_{false};

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* ABLWALLFRICTIONVELALG_H */
