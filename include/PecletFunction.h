/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef PecletFunction_h
#define PecletFunction_h

#include "KokkosInterface.h"

namespace sierra{
namespace nalu{

template<typename T>
class PecletFunction
{
public:
  KOKKOS_FUNCTION PecletFunction() = default;
  KOKKOS_FUNCTION virtual ~PecletFunction() = default;
  KOKKOS_FUNCTION virtual T execute(const T pecletNumber) = 0;
};

template<typename T>
class ClassicPecletFunction : public PecletFunction<T>
{
public:
  KOKKOS_FUNCTION ClassicPecletFunction(T A, T hf);
  KOKKOS_FUNCTION virtual ~ClassicPecletFunction() = default;
  KOKKOS_FUNCTION T execute(const T pecletNumber);

  T A_;
  T hf_;
};

template<typename T>
class TanhFunction : public PecletFunction<T>
{
public:
  KOKKOS_FUNCTION TanhFunction( T c1, T c2 );
  KOKKOS_FUNCTION virtual ~TanhFunction() = default;
  KOKKOS_FUNCTION T execute(const T indVar);

  T c1_; // peclet number at which transition occurs
  T c2_; // width of the transtion
};

} // namespace nalu
} // namespace Sierra

#endif
