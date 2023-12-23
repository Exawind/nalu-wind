// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VIEWHELPER_H
#define VIEWHELPER_H

#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "ScratchViews.h"

namespace sierra {
namespace nalu {

namespace nalu_ngp {

template <typename ElemSimdDataType, typename ScalarFieldType>
struct ScalarViewHelper
{
  using SimdDataType = ElemSimdDataType;
  using ViewDataType =
    SharedMemView<DoubleType*, typename ElemSimdDataType::ShmemType>;
  using ScratchViewsType = ScratchViews<
    DoubleType,
    typename ElemSimdDataType::TeamHandleType,
    typename ElemSimdDataType::ShmemType>;

  KOKKOS_INLINE_FUNCTION
  ScalarViewHelper(ScratchViewsType& scrView, unsigned phiID)
    : v_phi_(scrView.get_scratch_view_1D(phiID))
  {
  }

  KOKKOS_DEFAULTED_FUNCTION
  ~ScalarViewHelper() = default;

  KOKKOS_INLINE_FUNCTION
  DoubleType operator()(int ni, int) const { return v_phi_(ni); }

  const ViewDataType& v_phi_;
};

template <typename ElemSimdDataType, typename VectorFieldType>
struct VectorViewHelper
{
  using SimdDataType = ElemSimdDataType;
  using ViewDataType =
    SharedMemView<DoubleType**, typename ElemSimdDataType::ShmemType>;
  using ScratchViewsType = ScratchViews<
    DoubleType,
    typename ElemSimdDataType::TeamHandleType,
    typename ElemSimdDataType::ShmemType>;

  KOKKOS_INLINE_FUNCTION
  VectorViewHelper(ScratchViewsType& scrView, unsigned phiID)
    : v_phi_(scrView.get_scratch_view_2D(phiID))
  {
  }

  KOKKOS_DEFAULTED_FUNCTION
  ~VectorViewHelper() = default;

  KOKKOS_INLINE_FUNCTION
  DoubleType operator()(int ni, int ic) const { return v_phi_(ni, ic); }

  const ViewDataType& v_phi_;
};

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* VIEWHELPER_H */
