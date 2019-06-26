/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
