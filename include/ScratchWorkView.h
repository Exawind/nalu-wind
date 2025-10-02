// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ScratchWorkView_h
#define ScratchWorkView_h

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

template <int n, typename ViewType>
struct ScratchWorkView
{
  static constexpr int length = n;
  using view_type = ViewType;
  using value_type = typename view_type::value_type;

  ScratchWorkView() = default;

  explicit ScratchWorkView(value_type init_val)
  {
    for (int j = 0; j < n; ++j)
      data_[j] = init_val;
  }

  ViewType& view() { return view_; }
  const ViewType& view() const { return view_; }
  value_type* data() { return data_.data(); }
  const value_type* data() const { return data_.data(); }

   Kokkos::Array<value_type, n> data_{};
  ViewType view_{data_.data()};
};

} // namespace nalu
} // namespace sierra

#endif
