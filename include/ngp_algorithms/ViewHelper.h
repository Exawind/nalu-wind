/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef VIEWHELPER_H
#define VIEWHELPER_H

#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "ScratchViews.h"

namespace sierra {
namespace nalu {

namespace nalu_ngp {

template<typename ElemSimdDataType, typename FieldType>
struct ViewHelper;

template<typename ElemSimdDataType>
struct ViewHelper<ElemSimdDataType, ScalarFieldType>
{
  using ViewDataType = SharedMemView<DoubleType*,
                                     typename ElemSimdDataType::ShmemType>;
  using ScratchViewsType = ScratchViews<DoubleType,
                                        typename ElemSimdDataType::TeamHandleType,
                                        typename ElemSimdDataType::ShmemType>;

  KOKKOS_INLINE_FUNCTION
  ViewHelper(ScratchViewsType& scrView, unsigned phiID)
    : v_phi_(scrView.get_scratch_view_1D(phiID))
  {}

  KOKKOS_FUNCTION
  ~ViewHelper() = default;

  KOKKOS_INLINE_FUNCTION
  DoubleType operator()(int ni, int) const
  { return v_phi_(ni); }

  const ViewDataType& v_phi_;
};

template<typename ElemSimdDataType>
struct ViewHelper<ElemSimdDataType, VectorFieldType>
{
  using ViewDataType = SharedMemView<DoubleType**,
                                     typename ElemSimdDataType::ShmemType>;
  using ScratchViewsType = ScratchViews<DoubleType,
                                        typename ElemSimdDataType::TeamHandleType,
                                        typename ElemSimdDataType::ShmemType>;

  KOKKOS_INLINE_FUNCTION
  ViewHelper(ScratchViewsType& scrView, unsigned phiID)
    : v_phi_(scrView.get_scratch_view_2D(phiID))
  {}

  KOKKOS_FUNCTION
  ~ViewHelper() = default;

  KOKKOS_INLINE_FUNCTION
  DoubleType operator()(int ni, int ic) const
  { return v_phi_(ni, ic); }

  const ViewDataType& v_phi_;
};

}  // nalu_ngp
}  // nalu
}  // sierra


#endif /* VIEWHELPER_H */
