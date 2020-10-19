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

/** Simultaneously compute the wall shear stress and heat flux/surface
 *  temperature.  This boundary condition follows the algorithms outlined
 *  in
 *
 *     Basu, S., Holtslag, A. A. M., Bas, J. H., Van de Wiel, B. J. H., 
 *     Moene, A. F., and Steeneveld, G.-J., An Inconvenient "Truth" About
 *     Using Sensible Heat Flux As A Surface Boundary Condition in Models.
 *     Acta Geophysica, 56(1), pp. 88-99. 
 *     https://doi.org/10.2478/s11600-007-0038-y
 *
 *  The velocity and temperature are sampled at the nodes opposing the
 *  wall nodes with the option for some type of averaging.  Then those
 *  values are used in the Monin-Obukhov similarity laws to compute
 *  friction velocity and heat flux.  Friction velocity is then used to
 *  compute wall shear stress with directionality related to the opposing
 *  node velocity vector.  Wall shear stress and heat flux can then be
 *  made locally fluctuating.
 * 
 *  An example from the input file is:
 *
 *```
 *    - wall_boundary_condition: bc_lower
 *      target_name: lower
 *       wall_user_data:
 *         velocity: [0.0,0.0,0.0]
 *         abl_wall_function:
 *           surface_heating_table:
 *             - [     0.0, 0.00, 300.0, 1.0]
 *             - [999999.9, 0.00, 300.0, 1.0]
 *           reference_temperature: 300.0
 *           roughness_height: 0.1
 *           kappa: 0.4
 *           beta_m: 5.0
 *           beta_h: 5.0
 *           gamma_m: 16.0
 *           gamma_h: 16.0
 *           gravity_vector_component: 3
 *           monin_obukhov_averaging_type: planar
 *           fluctuation_model: Moeng
 *           fluctuating_temperature_ref: surface
 *``
 *  
 *
 *  /sa WallFricVelAlgDriver, BdyLayerStatistics, WallFuncGeometryAlg  
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
  ElemDataRequests elemData_;

  unsigned coordinates_     {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_     {stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_      {stk::mesh::InvalidOrdinal};
  unsigned temperatureNp1_  {stk::mesh::InvalidOrdinal};
  unsigned density_         {stk::mesh::InvalidOrdinal};
  unsigned bcHeatFlux_      {stk::mesh::InvalidOrdinal};
  unsigned wallHeatFlux_    {stk::mesh::InvalidOrdinal};
  unsigned specificHeat_    {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_  {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_     {stk::mesh::InvalidOrdinal};
  unsigned wallShearStress_ {stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_    {stk::mesh::InvalidOrdinal};

  //! Break the flux/surface temperature vs. time input table into vectors
  //! of each quantity and store in the following vectors.
  std::vector<DblType> tableTimes_{0.0,999999.9};
  std::vector<DblType> tableFluxes_{0.0,0.0};
  std::vector<DblType> tableSurfaceTemperatures_{301.0, 301.0};
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
  std::string averagingType_{"planar"};

  //! The model for applying fluctuations to the lower shear stress and heat flux.
  //! Current options are:
  //!   - none - Apply no fluctuations--use the base fluxes.
  //!   - Schumann - Use the model of Schumann.
  //!   - Moeng - Use the model of Moeng.
  //! Future option that should be tried are:
  //!   - Brasseur - Brasseur's modification to Moeng.
  std::string fluctuationModel_{"Schumann"};


  //! In computing the fluctuating temperature flux, the difference between the
  //! planar averaged or local temperature at height z1 and some reference temperature
  //! is taken.  Moeng does not really define that reference temperature.  We find that
  //! if it is taken as TRef, and the simulation is such that the z1 temperature crosses
  //! Tref, the fluctuations get very large during that crossover, which is unphysical.
  //! Therefore we add the option of using current surface temperature as this reference
  //! that is subtracted from the z1 temperature.  
  //! Current options are:
  //!   - surface - Use current surface temperature (time varying) as the reference.
  //!   - reference - Use the reference temperature (time invariant) as the reference.
  std::string fluctuatingTempRef_{"surface"};

  //! Monin-Obukhov scaling law constants.
  //! These should really be variable given stability, but they are just fixed for now.
  DblType kappa_{0.41};
  DblType beta_m_{5.0};
  DblType beta_h_{5.0};
  DblType gamma_m_{16.0};
  DblType gamma_h_{16.0};

  bool useShifted_{false};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* ABLWALLFLUXESALG_H */
