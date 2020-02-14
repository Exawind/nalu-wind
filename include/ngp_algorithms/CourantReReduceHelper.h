// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANTREREDUCEHELPER_H
#define COURANTREREDUCEHELPER_H

#include "KokkosInterface.h"

namespace sierra {
namespace nalu {

struct CflRe
{
  double max_cfl, max_re;

  KOKKOS_INLINE_FUNCTION
  CflRe()
  {
    max_cfl = -1.0e6;
    max_re = -1.0e6;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const CflRe& rhs)
  {
    max_cfl = rhs.max_cfl;
    max_re = rhs.max_re;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const volatile CflRe& rhs) volatile
  {
    max_cfl = rhs.max_cfl;
    max_re = rhs.max_re;
  }
};

template<typename Space=Kokkos::HostSpace>
struct CflReMax
{
public:
  typedef CflReMax reducer;
  typedef CflRe value_type;
  typedef Kokkos::View<value_type, Space> result_view_type;

private:
  result_view_type value;
  bool references_scalar_v;

public:
  KOKKOS_INLINE_FUNCTION
  CflReMax(value_type& value_): value(&value_),references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  CflReMax(const result_view_type& value_): value(value_),references_scalar_v(false) {}

  //Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src)  const {
    if (dest.max_cfl < src.max_cfl) dest.max_cfl = src.max_cfl;
    if (dest.max_re < src.max_re) dest.max_re = src.max_re;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    if (dest.max_cfl < src.max_cfl) dest.max_cfl = src.max_cfl;
    if (dest.max_re < src.max_re) dest.max_re = src.max_re;
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type& val)  const {
    val.max_cfl = Kokkos::reduction_identity<double>::max();;
    val.max_re = Kokkos::reduction_identity<double>::max();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return *value.data();
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const {
    return references_scalar_v;
  }
};

}  // nalu
}  // sierra


#endif /* COURANTREREDUCEHELPER_H */
