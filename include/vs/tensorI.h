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

#ifndef VS_TENSORI_H
#define VS_TENSORI_H

#include <cmath>
#include "vs/tensor.h"
#include "vs/trig_ops.h"

namespace vs {

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
TensorT<T>::TensorT(
  const VectorT<T>& x,
  const VectorT<T>& y,
  const VectorT<T>& z,
  const bool transpose)
{
  if (transpose) {
    this->cols(x, y, z);
  } else {
    this->rows(x, y, z);
  }
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void
TensorT<T>::rows(
  const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept
{
  // clang-format off
    vv[0] = x.x(); vv[1] = x.y(); vv[2] = x.z();
    vv[3] = y.x(); vv[4] = y.y(); vv[5] = y.z();
    vv[6] = z.x(); vv[7] = z.y(); vv[8] = z.z();
  // clang-format on
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void
TensorT<T>::cols(
  const VectorT<T>& x, const VectorT<T>& y, const VectorT<T>& z) noexcept
{
  // clang-format off
    vv[0] = x.x(); vv[1] = y.x(); vv[2] = z.x();
    vv[3] = x.y(); vv[4] = y.y(); vv[5] = z.y();
    vv[6] = x.z(); vv[7] = y.z(); vv[8] = z.z();
  // clang-format on
}

// clang-format off
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::x() const noexcept
{
    return VectorT<T>{vv[0], vv[1], vv[2]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::y() const noexcept
{
    return VectorT<T>{vv[3], vv[4], vv[5]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::z() const noexcept
{
    return VectorT<T>{vv[6], vv[7], vv[8]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::cx() const noexcept
{
    return VectorT<T>{vv[0], vv[3], vv[6]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::cy() const noexcept
{
    return VectorT<T>{vv[1], vv[4], vv[7]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION
VectorT<T> TensorT<T>::cz() const noexcept
{
    return VectorT<T>{vv[2], vv[5], vv[8]};
}
// clang-format on

template <typename T, typename OStream>
OStream&
operator<<(OStream& out, const TensorT<T>& t)
{
  out << "(";
  for (int i = 0; i < t.ncomp; ++i) {
    out << " " << t[i];
  }
  out << " )";
  return out;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator&(const TensorT<T>& t, const VectorT<T>& v)
{
  VectorT<T> out;
  out.x() = t.xx() * v.x() + t.xy() * v.y() + t.xz() * v.z();
  out.y() = t.yx() * v.x() + t.yy() * v.y() + t.yz() * v.z();
  out.z() = t.zx() * v.x() + t.zy() * v.y() + t.zz() * v.z();
  return out;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION VectorT<T>
operator&(const VectorT<T>& v, const TensorT<T>& t)
{
  VectorT<T> out;
  out.x() = t.xx() * v.x() + t.yx() * v.y() + t.zx() * v.z();
  out.y() = t.xy() * v.x() + t.yy() * v.y() + t.zy() * v.z();
  out.z() = t.xz() * v.x() + t.yz() * v.y() + t.zz() * v.z();
  return out;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION TensorT<T>
operator&(const TensorT<T>& t1, const TensorT<T>& t2)
{
  TensorT<T> t3;

  t3.xx() = t1.xx() * t2.xx() + t1.xy() * t2.yx() + t1.xz() * t2.zx();
  t3.xy() = t1.xx() * t2.xy() + t1.xy() * t2.yy() + t1.xz() * t2.zy();
  t3.xz() = t1.xx() * t2.xz() + t1.xy() * t2.yz() + t1.xz() * t2.zz();

  t3.yx() = t1.yx() * t2.xx() + t1.yy() * t2.yx() + t1.yz() * t2.zx();
  t3.yy() = t1.yx() * t2.xy() + t1.yy() * t2.yy() + t1.yz() * t2.zy();
  t3.yz() = t1.yx() * t2.xz() + t1.yy() * t2.yz() + t1.yz() * t2.zz();

  t3.zx() = t1.zx() * t2.xx() + t1.zy() * t2.yx() + t1.zz() * t2.zx();
  t3.zy() = t1.zx() * t2.xy() + t1.zy() * t2.yy() + t1.zz() * t2.zy();
  t3.zz() = t1.zx() * t2.xz() + t1.zy() * t2.yz() + t1.zz() * t2.zz();

  return t3;
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION TensorT<T>
operator+(const TensorT<T>& t1, const TensorT<T>& t2)
{
  return TensorT<T>{
    t1.vv[0] + t2.vv[0], t1.vv[1] + t2.vv[1], t1.vv[2] + t2.vv[2],
    t1.vv[3] + t2.vv[3], t1.vv[4] + t2.vv[4], t1.vv[5] + t2.vv[5],
    t1.vv[6] + t2.vv[6], t1.vv[7] + t2.vv[7], t1.vv[8] + t2.vv[8]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION TensorT<T>
operator-(const TensorT<T>& t1, const TensorT<T>& t2)
{
  return TensorT<T>{
    t1.vv[0] - t2.vv[0], t1.vv[1] - t2.vv[1], t1.vv[2] - t2.vv[2],
    t1.vv[3] - t2.vv[3], t1.vv[4] - t2.vv[4], t1.vv[5] - t2.vv[5],
    t1.vv[6] - t2.vv[6], t1.vv[7] - t2.vv[7], t1.vv[8] - t2.vv[8]};
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
operator&&(const TensorT<T>& t1, const TensorT<T>& t2)
{
  return (
    t1.vv[0] * t2.vv[0] + t1.vv[1] * t2.vv[1] + t1.vv[2] * t2.vv[2] +
    t1.vv[3] * t2.vv[3] + t1.vv[4] * t2.vv[4] + t1.vv[5] * t2.vv[5] +
    t1.vv[6] * t2.vv[6] + t1.vv[7] * t2.vv[7] + t1.vv[8] * t2.vv[8]);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
mag_sqr(const TensorT<T>& t)
{
  // cppcheck-suppress duplicateExpression
  return (t && t);
}

template <typename T>
KOKKOS_FORCEINLINE_FUNCTION T
mag(const TensorT<T>& t)
{
  return std::sqrt(mag_sqr(t));
}

KOKKOS_FORCEINLINE_FUNCTION Tensor
xrot(const double angle)
{
  const double ang = utils::radians(angle);
  const double cval = std::cos(ang);
  const double sval = std::sin(ang);

  return Tensor{1.0, 0.0, 0.0, 0.0, cval, sval, 0.0, -sval, cval};
}

KOKKOS_FORCEINLINE_FUNCTION Tensor
yrot(const double angle)
{
  const double ang = utils::radians(angle);
  const double cval = std::cos(ang);
  const double sval = std::sin(ang);

  return Tensor{cval, 0.0, -sval, 0.0, 1.0, 0.0, sval, 0.0, cval};
}

KOKKOS_FORCEINLINE_FUNCTION Tensor
zrot(const double angle)
{
  const double ang = utils::radians(angle);
  const double cval = std::cos(ang);
  const double sval = std::sin(ang);

  return Tensor{cval, sval, 0.0, -sval, cval, 0.0, 0.0, 0.0, 1.0};
}

KOKKOS_FORCEINLINE_FUNCTION Tensor
quaternion(const Vector& axis, const double angle)
{
  const double ang = -1.0 * utils::radians(angle);
  const double cval = std::cos(0.5 * ang);
  const double sval = std::sin(0.5 * ang);
  const auto vmag = mag(axis);
  const double q0 = cval;
  const double q1 = sval * axis.x() / vmag;
  const double q2 = sval * axis.y() / vmag;
  const double q3 = sval * axis.z() / vmag;

  Tensor t;
  t.xx() = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
  t.xy() = 2.0 * (q1 * q2 - q0 * q3);
  t.xz() = 2.0 * (q0 * q2 + q1 * q3);

  t.yx() = 2.0 * (q1 * q2 + q0 * q3);
  t.yy() = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
  t.yz() = 2.0 * (q2 * q3 - q0 * q1);

  t.zx() = 2.0 * (q1 * q3 - q0 * q2);
  t.zy() = 2.0 * (q0 * q1 + q2 * q3);
  t.zz() = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

  return t;
}

} // namespace vs

#endif /* VS_TENSORI_H */
