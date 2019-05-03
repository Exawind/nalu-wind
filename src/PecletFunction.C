/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <PecletFunction.h>
#include "SimdInterface.h"

// basic c++
#include <algorithm>
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ClassicPecletFunction - classic
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template<typename T>
ClassicPecletFunction<T>::ClassicPecletFunction( const T A, const T hf)
  : A_(A),
    hf_(hf)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template<typename T>
T ClassicPecletFunction<T>::execute(const T pecletNumber)
{
  const T modPeclet = hf_*pecletNumber;
  return modPeclet*modPeclet/(5.0 + modPeclet*modPeclet);
}

//==========================================================================
// Class Definition
//==========================================================================
// TanhFunction - classic
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
template<typename T>
TanhFunction<T>::TanhFunction(T c1, T c2)
  : c1_(c1),
    c2_(c2)
{
  // nothing to do; assume that the functional form varies between 0 and 1
}


//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template<typename T>
T TanhFunction<T>::execute(const T indVar)
{
  return 0.50*(1.0+stk::math::tanh((indVar-c1_)/c2_));
}

template class ClassicPecletFunction<double>;
template class TanhFunction<double>;

#ifdef STK_HAVE_SIMD
template class ClassicPecletFunction<DoubleType>;
template class TanhFunction<DoubleType>;
#endif

} // namespace nalu
} // namespace Sierra
