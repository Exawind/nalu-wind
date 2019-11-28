// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef PecletFunction_h
#define PecletFunction_h

#include "KokkosInterface.h"

namespace sierra{
namespace nalu{

/** Non-templated empty base class for storing pointers to templated instances
 */
class PecletFunctionBase
{};

template<typename T>
class PecletFunction : public PecletFunctionBase
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
