// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
// Original implementation of this code by Shreyas Ananthan for AMR-Wind
// - (https://github.com/Exawind/amr-wind)
//
// Adapted to use Kokkos

#ifndef VS_VECTORI_H
#define VS_VECTORI_H

#include <ostream>
#include <cmath>
#include "vs/vector.h"

namespace vs {

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
VectorT<T>::operator-() const
{
  return VectorT<T>{-vv[0], -vv[1], -vv[2]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
VectorT<T>::operator*=(const T fac)
{
  vv[0] *= fac;
  vv[1] *= fac;
  vv[2] *= fac;
  return *this;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
VectorT<T>::operator/=(const T fac)
{
  vv[0] /= fac;
  vv[1] /= fac;
  vv[2] /= fac;
  return *this;
}

template <typename T, typename OStream>
OStream&
operator<<(OStream& out, const VectorT<T>& vec)
{
  out << "(" << vec.x() << " " << vec.y() << " " << vec.z() << ")";
  return out;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator+(const VectorT<T>& v1, const VectorT<T>& v2)
{
  return VectorT<T>{v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator-(const VectorT<T>& v1, const VectorT<T>& v2)
{
  return VectorT<T>{v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator*(const VectorT<T1>& inp, const T2 fac)
{
  return VectorT<T1>{inp.x() * fac, inp.y() * fac, inp.z() * fac};
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator*(const T2 fac, const VectorT<T1>& inp)
{
  return VectorT<T1>{inp.x() * fac, inp.y() * fac, inp.z() * fac};
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator/(const VectorT<T1>& inp, const T2 fac)
{
  return VectorT<T1>{inp.x() / fac, inp.y() / fac, inp.z() / fac};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
operator&(const VectorT<T>& v1, const VectorT<T>& v2)
{
  return (v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z());
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator^(const VectorT<T>& v1, const VectorT<T>& v2)
{
  return VectorT<T>{
    (v1.y() * v2.z() - v1.z() * v2.y()), (v1.z() * v2.x() - v1.x() * v2.z()),
    (v1.x() * v2.y() - v1.y() * v2.x())};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator*(const VectorT<T>& v1, const VectorT<T>& v2)
{
  return VectorT<T>{v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z()};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
mag_sqr(const VectorT<T>& v)
{
  return (v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
mag(const VectorT<T>& v)
{
  return std::sqrt(mag_sqr(v));
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>&
VectorT<T>::normalize()
{
  T vmag = mag(*this);

  if (vmag < Traits::eps()) {
    *this = VectorT<T>::zero();
  } else {
    *this /= vmag;
  }
  return *this;
}

#if 0
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator+(VectorT<T>&& v1, const VectorT<T>& v2)
{
    v1.x() += v2.x();
    v1.y() += v2.y();
    v1.z() += v2.z();
    return v1;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator+(const VectorT<T>& v2, VectorT<T>&& v1)
{
    v1.x() += v2.x();
    v1.y() += v2.y();
    v1.z() += v2.z();
    return v1;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator+(VectorT<T>&& v1, VectorT<T>&& v2)
{
    v1.x() += v2.x();
    v1.y() += v2.y();
    v1.z() += v2.z();
    return v1;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator-(VectorT<T>&& v1, const VectorT<T>& v2)
{
    v1.x() -= v2.x();
    v1.y() -= v2.y();
    v1.z() -= v2.z();
    return v1;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator-(const VectorT<T>& v2, VectorT<T>&& v1)
{
    v1.x() -= v2.x();
    v1.y() -= v2.y();
    v1.z() -= v2.z();
    return v1;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator-(VectorT<T>&& v1, VectorT<T>&& v2)
{
    v1.x() -= v2.x();
    v1.y() -= v2.y();
    v1.z() -= v2.z();
    return v1;
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator*(VectorT<T1>&& inp, const T2 fac)
{
    inp.x() *= fac;
    inp.y() *= fac;
    inp.z() *= fac;
    return inp;
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator*(const T2 fac, VectorT<T1>&& inp)
{
    inp.x() *= fac;
    inp.y() *= fac;
    inp.z() *= fac;
    return inp;
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator/(VectorT<T1>&& inp, const T2 fac)
{
    inp.x() /= fac;
    inp.y() /= fac;
    inp.z() /= fac;
    return inp;
}

template <typename T1, typename T2>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T1>
operator/(const T2 fac, VectorT<T1>&& inp)
{
    inp.x() /= fac;
    inp.y() /= fac;
    inp.z() /= fac;
    return inp;
}

#endif
} // namespace vs

#endif /* VS_VECTORI_H */
