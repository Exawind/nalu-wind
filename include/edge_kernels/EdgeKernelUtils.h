// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef EDGEKERNELUTILS_H
#define EDGEKERNELUTILS_H

#include "SimdInterface.h"

namespace sierra {
namespace nalu {

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
T van_leer(const T& dqm, const T& dqp, const T& eps)
{
  return (2.0 * (dqm * dqp + stk::math::abs(dqm * dqp))) /
         ((dqm + dqp) * (dqm + dqp) + eps);
}

}  // nalu
}  // sierra


#endif /* EDGEKERNELUTILS_H */
