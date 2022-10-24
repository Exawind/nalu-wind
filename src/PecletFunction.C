// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <PecletFunction.h>
#include "SimdInterface.h"

// basic c++
#include <algorithm>
#include <cmath>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ClassicPecletFunction - classic
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename T>
KOKKOS_FUNCTION
ClassicPecletFunction<T>::ClassicPecletFunction(const T A, const T hf)
  : A_(A), hf_(hf)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template <typename T>
KOKKOS_FUNCTION T
ClassicPecletFunction<T>::execute(const T pecletNumber)
{
  const T modPeclet = hf_ * pecletNumber;
  return modPeclet * modPeclet / (5.0 + modPeclet * modPeclet);
}

//==========================================================================
// Class Definition
//==========================================================================
// TanhFunction - classic
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template <typename T>
KOKKOS_FUNCTION
TanhFunction<T>::TanhFunction(T c1, T c2)
  : c1_(c1), c2_(c2)
{
  // nothing to do; assume that the functional form varies between 0 and 1
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template <typename T>
KOKKOS_FUNCTION T
TanhFunction<T>::execute(const T indVar)
{
  return 0.50 * (1.0 + stk::math::tanh((indVar - c1_) / c2_));
}

template class ClassicPecletFunction<double>;
template class TanhFunction<double>;

#ifdef STK_HAVE_SIMD
template class ClassicPecletFunction<DoubleType>;
template class TanhFunction<DoubleType>;
#endif

} // namespace nalu
} // namespace sierra
