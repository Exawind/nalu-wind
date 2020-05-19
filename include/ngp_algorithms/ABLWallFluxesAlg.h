// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ABLWALLFLUXESALG_H
#define ABLWALLFLUXESALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "ngp_algorithms/WallFricVelAlgDriver.h"

#include "stk_mesh/base/Types.hpp"

#include "NaluParsing.h"

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
class ABLWallFluxesAlg : public Algorithm
{
public:
  template <typename T>
  using ListArray = std::vector<std::vector<T>>;

  using DblType = double;

  ABLWallFluxesAlg(
    Realm&,
    stk::mesh::Part*,
    WallFricVelAlgDriver&,
    const bool,
    const YAML::Node&);

  virtual ~ABLWallFluxesAlg() = default;

  //! Read the input file to get information about the ABL boundary condition.
  void load(const YAML::Node&);

  virtual void execute() override;

private:
  WallFricVelAlgDriver& algDriver_;

  ElemDataRequests faceData_;

  unsigned velocityNp1_     {stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_      {stk::mesh::InvalidOrdinal};
  unsigned density_         {stk::mesh::InvalidOrdinal};
  unsigned bcHeatFlux_      {stk::mesh::InvalidOrdinal};
  unsigned wallHeatFlux_    {stk::mesh::InvalidOrdinal};
  unsigned specificHeat_    {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_  {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_     {stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_    {stk::mesh::InvalidOrdinal};

  // Break the flux/surface temperature vs. time input table into vectors
  // of each quantity and store in the following vectors.
  std::vector<DblType> tableTimes_{0.0,999999.9};
  std::vector<DblType> tableFluxes_{0.0,0.0};
  std::vector<DblType> tableSurfaceTemperatures_{Tref_,Tref_};
  std::vector<DblType> tableWeights_{0.0,0.0};

  //! Acceleration due to gravity (m/s^2)
  int gravityVectorComponent_{3};
  DblType gravity_{9.7};

  //! Roughness height (m)
  DblType z0_{0.001};

  //! Reference temperature (K)
  DblType Tref_{301.0};

  //! The type of averaging to apply to the Monin-Obukhov scaling law.
  //! Current options are:
  //!   - none - Apply no averaging--treat all quantities locally.
  //!   - planar - Apply planar averaging at the nodes adjacent to the wall nodes.
  //! Future options that should be tried are:
  //!   - time - Apply local time-averaging within some backward-in-time windows.
  //!   - Lagrangian - Apply Lagrangian averaging backward along a streamline.
  std::string averagingType_{"none"};

  //! Monin-Obukhov scaling law constants.
  //! These should really be variable given stability, but they are just fixed for now.
  DblType kappa_{0.41};
  DblType beta_m_{5.0};
  DblType beta_h_{5.0};
  DblType gamma_m_{16.0};
  DblType gamma_h_{16.0};

  bool useShifted_{false};

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* ABLWALLFLUXESALG_H */
