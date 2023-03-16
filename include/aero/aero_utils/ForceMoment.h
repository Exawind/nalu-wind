// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FORCE_MOMENT_H
#define FORCE_MOMENT_H

#include <KokkosInterface.h>

namespace fsi {

//! Compute the effective force and moment at the OpenFAST mesh node for a given
//! force at the CFD surface mesh node
void KOKKOS_INLINE_FUNCTION
computeEffForceMoment(
  double* forceCFD, double* xyzCFD, double* forceMomOF, double* xyzOF)
{

  const int ndim = 3; // I don't see this ever being used in other situations
  for (int j = 0; j < ndim; j++)
    forceMomOF[j] += forceCFD[j];
  forceMomOF[3] +=
    (xyzCFD[1] - xyzOF[1]) * forceCFD[2] - (xyzCFD[2] - xyzOF[2]) * forceCFD[1];
  forceMomOF[4] +=
    (xyzCFD[2] - xyzOF[2]) * forceCFD[0] - (xyzCFD[0] - xyzOF[0]) * forceCFD[2];
  forceMomOF[5] +=
    (xyzCFD[0] - xyzOF[0]) * forceCFD[1] - (xyzCFD[1] - xyzOF[1]) * forceCFD[0];
}

//! Split a force and moment into the surrounding 'left' and 'right' nodes in a
//! variationally consistent manner using
void KOKKOS_INLINE_FUNCTION
splitForceMoment(
  double* totForceMoment,
  double interpFac,
  double* leftForceMoment,
  double* rightForceMoment)
{
  for (int i = 0; i < 6; i++) {
    leftForceMoment[i] += (1.0 - interpFac) * totForceMoment[i];
    rightForceMoment[i] += interpFac * totForceMoment[i];
  }
}

} // namespace fsi

#endif
