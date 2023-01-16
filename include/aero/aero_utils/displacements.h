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
  // Kind of dangeraous constructor
  SixDOF(double* vec)
    : translation_({vec[0], vec[1], vec[2]}),
      rotation_({vec[3], vec[4], vec[5]})
  {
  }

  SixDOF(vs::Vector transDisp, vs::Vector rotDisp)
    : translation_(transDisp), rotation_(rotDisp)
  {
  }

  vs::Vector translation_;
  vs::Vector rotation_;
};

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
operator+(const SixDOF& a, const SixDOF& b)
{
  // adding b to a, so pushing b wmp onto a stack
  return SixDOF(
    a.translation_ + b.translation_, wmp::push(b.rotation_, a.rotation_));
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
operator-(const SixDOF& a, const SixDOF& b)
{
  // subtracting b from a, so poping b from the a stack
  return SixDOF(
    a.translation_ - b.translation_, wmp::pop(b.rotation_, a.rotation_));
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
linear_interp_total_displacement(
  const SixDOF start, const SixDOF end, const double interpFactor)
{
  auto transDisp = wmp::linear_interp_translation(
    start.translation_, end.translation_, interpFactor);
  auto rotDisp =
    wmp::linear_interp_rotation(start.rotation_, end.rotation_, interpFactor);
  return SixDOF(transDisp, rotDisp);
}

KOKKOS_FORCEINLINE_FUNCTION
SixDOF
linear_interp_total_velocity(
  const SixDOF start, const SixDOF end, const double interpFactor)
{
  auto transDisp = wmp::linear_interp_translation(
    start.translation_, end.translation_, interpFactor);
  auto rotDisp = wmp::linear_interp_translation(
    start.translation_, end.translation_, interpFactor);
  return SixDOF(transDisp, rotDisp);
}

//! Convert a position relative to an aerodynamic point to the intertial
//! coordinate system
KOKKOS_FORCEINLINE_FUNCTION
vs::Vector
local_aero_coordinates(
  const vs::Vector inertialPos, const SixDOF aeroRefPosition)
{
  const auto shift = inertialPos - aeroRefPosition.translation_;
  return wmp::rotate(aeroRefPosition.rotation_, shift);
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
  const auto delta = cfdPos - referencePos.translation_;
  // deflection roations need to be applied from the aerodynamic local frame of
  // reference
  const vs::Vector rotation =
    wmp::rotate(deflections.rotation_, localPos, true);
  return deflections.translation_ + rotation - delta;
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
  const vs::Vector /*root*/,
  const double /*pitch*/,
  const double /*rLoc*/)
{
  return compute_translational_displacements(deflections, referencePos, cfdPos);
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
  const auto pointRotate = wmp::rotate(totalDis.rotation_, pointLocal);
  return totalVel.translation_ + (totalVel.rotation_ ^ pointRotate);
}

} // namespace aero

#endif
