// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DISPLACEMENTS_H_
#define DISPLACEMENTS_H_

#include "vs/vector.h"
#include <Kokkos_Macros.hpp>
#include <aero/aero_utils/WienerMilenkovic.h>

namespace aero {
//! A struct to capture six degrees of freedom with a rotation and translation
//! components called out as separate entities the rotations are expressed as
//! WienerMilenkovic parameter
struct SixDOF
{

  SixDOF()
    : position_(0.0, 0.0, 0.0),
      orientation_(0.0, 0.0, 0.0)
  {
  }
    
  // Kind of dangerous constructor
  SixDOF(double* vec)
    : position_({vec[0], vec[1], vec[2]}),
      orientation_({vec[3], vec[4], vec[5]})
  {
  }

  SixDOF(vs::Vector transDisp, vs::Vector rotDisp)
    : position_(transDisp), orientation_(rotDisp)
  {
  }

  vs::Vector position_;
  vs::Vector orientation_;
};

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
operator+(const SixDOF& a, const SixDOF& b)
{
  // adding b to a, so pushing b wmp onto a stack
  return SixDOF(
    a.position_ + b.position_, wmp::push(b.orientation_, a.orientation_));
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
operator-(const SixDOF& a, const SixDOF& b)
{
  // subtracting b from a, so popping b from the a stack
  return SixDOF(
    a.position_ - b.position_, wmp::pop(b.orientation_, a.orientation_));
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
linear_interp_total_displacement(
  const SixDOF start, const SixDOF end, const double interpFactor)
{
  auto transDisp = wmp::linear_interp_translation(
    start.position_, end.position_, interpFactor);
  auto rotDisp = wmp::linear_interp_rotation(
    start.orientation_, end.orientation_, interpFactor);
  return SixDOF(transDisp, rotDisp);
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
linear_interp_total_velocity(
  const SixDOF start, const SixDOF end, const double interpFactor)
{
  auto transDisp = wmp::linear_interp_translation(
    start.position_, end.position_, interpFactor);
  auto rotDisp = wmp::linear_interp_translation(
    start.position_, end.position_, interpFactor);
  return SixDOF(transDisp, rotDisp);
}

//! Convert a position relative to an aerodynamic point to the intertial
//! coordinate system
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
local_aero_coordinates(
  const vs::Vector inertialPos, const SixDOF aeroRefPosition)
{
  const auto shift = inertialPos - aeroRefPosition.position_;
  return wmp::rotate(aeroRefPosition.orientation_, shift);
}

// TODO(psakiev) test this
//! Translate coordinate system for SixDOF variable from inertial to a reference
//! coordinate system
KOKKOS_FORCEINLINE_FUNCTION
SixDOF
local_aero_transformation(const SixDOF& inertialPos, const SixDOF& refPos)
{
  return SixDOF(
    local_aero_coordinates(inertialPos.position_, refPos),
    wmp::push(refPos.orientation_, inertialPos.orientation_));
}

//! Convert one array of 6 deflections (transX, transY, transZ, wmX, wmY,
//! wmZ) into one vector of translational displacement at a given node on the
//! turbine surface CFD mesh.
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compute_translational_displacements(
  const SixDOF deflections, const SixDOF referencePos, const vs::Vector cfdPos)
{
  const auto localPos = local_aero_coordinates(cfdPos, referencePos);
  const auto delta = cfdPos - referencePos.position_;
  // deflection roations need to be applied from the aerodynamic local frame of
  // reference
  const vs::Vector rotation =
    wmp::rotate(deflections.orientation_, localPos, true);
  return deflections.position_ + rotation - delta;
}

// TODO(psakiev) this function is a place holder for when we need to add pitch
// in the next PR
//
//! Accounting for pitch, convert one array of 6 deflections (transX, transY,
//! transZ, wmX, wmY, wmZ) into one vector of translational displacement at a
//! given node on the turbine surface CFD mesh.
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compute_translational_displacements(
  const SixDOF deflections,
  const SixDOF referencePos,
  const vs::Vector cfdPos,
  const SixDOF hub_disp,
  const SixDOF hub_ref,
  const double wall_dist,
  const double rLoc)
{
  auto full_disp = compute_translational_displacements(deflections, referencePos, cfdPos);

  
  auto stiff_disp = compute_translational_displacements(hub_disp, hub_ref, cfdPos);
  
  auto rloc_ramp = 0.1 + 0.05*std::tanh(2.0*(wall_dist-5.0));
  auto ramp = (0.5 + 0.5*std::tanh(20.0*(rLoc/61.5 - rloc_ramp)) );
  //auto ramp = 0.0;
  
  return stiff_disp + ramp * (full_disp - stiff_disp);
}

//! Convert one array of 6 velocities (transX, transY, transZ, wmX, wmY, wmZ)
//! into one vector of translational velocity at a given node on the turbine
//! surface CFD mesh.
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
compute_mesh_velocity(
  const SixDOF totalVel,
  const SixDOF totalDis,
  const SixDOF referencePos,
  const vs::Vector cfdPos)
{
  const auto pointLocal = local_aero_coordinates(cfdPos, referencePos);
  const auto pointRotate = wmp::rotate(totalDis.orientation_, pointLocal);
  return totalVel.position_ + (totalVel.orientation_ ^ pointRotate);
}

} // namespace aero

#endif
