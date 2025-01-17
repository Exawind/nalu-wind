// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANT_REYNOLDS_H
#define COURANT_REYNOLDS_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p>
struct max_local_courant_t
{
  static double
  invoke(double dt, const_vector_view<p> xc, const_vector_view<p> vel);
};

template <int p>
struct max_local_reynolds_t
{
  static double invoke(
    const_vector_view<p> xc,
    const_scalar_view<p> rho,
    const_scalar_view<p> visc,
    const_vector_view<p> vel);
};

template <int p>
struct max_local_courant_reynolds_t
{
  static Kokkos::Array<double, 2> invoke(
    double dt,
    const_vector_view<p> xc,
    const_scalar_view<p> rho,
    const_scalar_view<p> visc,
    const_vector_view<p> vel);
};

} // namespace impl
P_INVOKEABLE(max_local_courant)
P_INVOKEABLE(max_local_reynolds)
P_INVOKEABLE(max_local_courant_reynolds)

template <
  typename BuiltinReducerFirst,
  typename BuiltinReducerSecond = BuiltinReducerFirst>
struct PairReduce
{
public:
  using reducer = PairReduce;
  using underlying_type_first = typename BuiltinReducerFirst::value_type;
  using underlying_type_second = typename BuiltinReducerSecond::value_type;
  using value_type =
    Kokkos::pair<underlying_type_first, underlying_type_second>;
  using result_view_type = Kokkos::View<
    value_type,
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::MemoryUnmanaged>;

  KOKKOS_INLINE_FUNCTION
  PairReduce(value_type& value_) : value(value_) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src) const
  {
    builtin_first.join(dst.first, src.first);
    builtin_second.join(dst.second, src.second);
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dst, const volatile value_type& src) const
  {
    builtin_first.join(dst.first, src.first);
    builtin_second.join(dst.second, src.second);
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const
  {
    builtin_first.init(val.first);
    builtin_second.init(val.second);
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value); }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }

private:
  value_type& value;

  underlying_type_first dummy_first{};
  BuiltinReducerFirst builtin_first{dummy_first};

  underlying_type_second dummy_second{};
  BuiltinReducerSecond builtin_second{dummy_second};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
