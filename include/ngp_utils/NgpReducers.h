// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPREDUCERS_H
#define NGPREDUCERS_H

#include "KokkosInterface.h"

namespace sierra {
namespace nalu {

namespace nalu_ngp {
template<class Scalar>
struct MinMaxSumScalar {
  Scalar min_val,max_val, total_sum;

  KOKKOS_INLINE_FUNCTION
  void operator = (const MinMaxSumScalar& rhs) {
    min_val = rhs.min_val;
    max_val = rhs.max_val;
    total_sum = rhs.total_sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator = (const volatile MinMaxSumScalar& rhs) volatile {
    min_val = rhs.min_val;
    max_val = rhs.max_val;
    total_sum = rhs.total_sum;
  }
};

template<class Scalar, class Space = Kokkos::HostSpace>
struct MinMaxSum
{
private:
  typedef typename std::remove_cv<Scalar>::type scalar_type;

public:
  typedef MinMaxSum reducer;
  typedef MinMaxSumScalar<scalar_type> value_type;
  typedef Kokkos::View<value_type, Space> result_view_type;

private:
  result_view_type value;
  bool references_scalar_v;

public:
  KOKKOS_INLINE_FUNCTION
  MinMaxSum(value_type& value_): value(&value_),references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  MinMaxSum(const result_view_type& value_): value(value_),references_scalar_v(false) {}

  //Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src)  const {
    if ( src.min_val < dest.min_val ) {
      dest.min_val = src.min_val;
    }
    if ( src.max_val > dest.max_val ) {
      dest.max_val = src.max_val;
    }

    dest.total_sum += src.total_sum;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    if ( src.min_val < dest.min_val ) {
      dest.min_val = src.min_val;
    }
    if ( src.max_val > dest.max_val ) {
      dest.max_val = src.max_val;
    }

    dest.total_sum += src.total_sum;
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type& val)  const {
    val.max_val = Kokkos::reduction_identity<scalar_type>::max();;
    val.min_val = Kokkos::reduction_identity<scalar_type>::min();
    val.total_sum = Kokkos::reduction_identity<scalar_type>::sum();
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

}

}  // nalu
}  // sierra



#endif /* NGPREDUCERS_H */
