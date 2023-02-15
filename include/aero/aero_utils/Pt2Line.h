// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PT2LINE_H
#define PT2LINE_H

#include <KokkosInterface.h>
#include <vs/vector.h>

namespace fsi {

/** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the
   non-dimensional location of the projected point along the line in [0-1]
   coordinates \f[ nonDimCoord = \frac{ (\vec{pt} - \vec{lStart}) \cdot (
   \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot
   (\vec{lEnd} - \vec{lStart}) } \f]
*/
KOKKOS_INLINE_FUNCTION
double
projectPt2Line(
  const vs::Vector& pt, const vs::Vector& lStart, const vs::Vector& lEnd)
{
  double nonDimCoord = 0.0;

  double num = 0.0;
  double denom = 0.0;

  for (int i = 0; i < 3; i++) {
    num += (pt[i] - lStart[i]) * (lEnd[i] - lStart[i]);
    denom += (lEnd[i] - lStart[i]) * (lEnd[i] - lStart[i]);
  }

  nonDimCoord = num / denom;
  return nonDimCoord;
}

/** Project a point 'pt' onto a line from 'lStart' to 'lEnd' and return the
   non-dimensional distance of 'pt' from the line w.r.t the distance from
   'lStart' to 'lEnd' \f[
    \vec{perp} &= (\vec{pt} - \vec{lStart}) - \frac{ (\vec{pt} - \vec{lStart})
   \cdot ( \vec{lEnd} - \vec{lStart} ) }{ (\vec{lEnd} - \vec{lStart}) \cdot
   (\vec{lEnd} - \vec{lStart}) } ( \vec{lEnd} - \vec{lStart} ) \\
    nonDimPerpDist = \frac{\lvert \vec{perp} \rvert}{ \lvert  (\vec{lEnd} -
   \vec{lStart}) \rvert } \f]
*/
KOKKOS_INLINE_FUNCTION
double
perpProjectDist_Pt2Line(
  const vs::Vector& pt, const vs::Vector& lStart, const vs::Vector& lEnd)
{
  double nonDimCoord = 0.0;
  double num = 0.0;
  double denom = 0.0;
  for (int i = 0; i < 3; i++) {
    num += (pt[i] - lStart[i]) * (lEnd[i] - lStart[i]);
    denom += (lEnd[i] - lStart[i]) * (lEnd[i] - lStart[i]);
  }
  nonDimCoord = num / denom;

  double nonDimPerpDist = 0.0;
  for (int i = 0; i < 3; i++) {
    double tmp = (pt[i] - lStart[i]) - nonDimCoord * (lEnd[i] - lStart[i]);
    nonDimPerpDist += tmp * tmp;
  }
  nonDimPerpDist = stk::math::sqrt(nonDimPerpDist / denom);

  return nonDimPerpDist;
}

} // namespace fsi

#endif
