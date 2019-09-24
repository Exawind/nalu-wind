/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ABLWALLFRICTIONVELALG_H
#define ABLWALLFRICTIONVELALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
class ABLWallFrictionVelAlg : public Algorithm
{
public:
  using DblType = double;

  ABLWallFrictionVelAlg(
    Realm&,
    stk::mesh::Part*,
    const bool,
    const double,
    const double,
    const double,
    const double);

  virtual ~ABLWallFrictionVelAlg() = default;

  virtual void execute() override;

private:
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
