/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ScratchWorkView_h
#define ScratchWorkView_h

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

template <int n, typename ViewType> struct ScratchWorkView
{
  static constexpr int length = n;
  using view_type = ViewType;
  using value_type = typename view_type::value_type;

  ScratchWorkView() = default;

  explicit ScratchWorkView(value_type init_val)
  {
    for (int j = 0; j < n; ++j) data_[j] = init_val;
  }

  ViewType& view() { return view_; }
  const ViewType& view() const {return view_; }
  value_type* data() { return data_.data();}
  const value_type* data() const {return data_.data();}

  NALU_ALIGNED Kokkos::Array<value_type, n> data_ {};
  ViewType view_{data_.data()};
};


} // namespace nalu
} // namespace Sierra

#endif
