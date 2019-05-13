/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef CopyAndInterleave_h
#define CopyAndInterleave_h

#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <ScratchViews.h>
#include <MultiDimViews.h>

namespace sierra{
namespace nalu{

template<typename SimdViewType, typename ViewType>
KOKKOS_FUNCTION
void interleave(SimdViewType& dview, const ViewType& sview, int simdIndex)
{
  int sz = dview.size();
  typename SimdViewType::pointer_type data = dview.data();
  typename ViewType::pointer_type src = sview.data();
  for(int i=0; i<sz; ++i) {
    stk::simd::set_data(data[i], simdIndex, src[i]);
  }
}

template<typename SimdViewType>
KOKKOS_FUNCTION
void interleave(SimdViewType& dview, const double* sviews[], int simdElems)
{
    int dim = dview.size();
    DoubleType* dptr = dview.data();
    for(int i=0; i<dim; ++i) {
        DoubleType& d = dptr[i];
        for(int simdIndex=0; simdIndex<simdElems; ++simdIndex) {
            stk::simd::set_data(d, simdIndex, sviews[simdIndex][i]);
        }
    }
}

template<typename MultiDimViewsType, typename SimdMultiDimViewsType>
KOKKOS_INLINE_FUNCTION
void copy_and_interleave(const MultiDimViewsType ** data,
                         int simdElems,
                         SimdMultiDimViewsType& simdData)
{
  const double* src[stk::simd::ndoubles] = {nullptr};
  unsigned numViews = simdData.get_num_1D_views();
  for(unsigned viewIndex=0; viewIndex<numViews; ++viewIndex) {
    for(int simdIndex=0; simdIndex<simdElems; ++simdIndex) {
      src[simdIndex] = data[simdIndex]->get_1D_view_by_index(viewIndex).data();
      NGP_ThrowAssert(data[simdIndex]->get_1D_view_by_index(viewIndex).size() == simdData.get_1D_view_by_index(viewIndex).size());
      NGP_ThrowAssert(src[simdIndex] != nullptr);
      NGP_ThrowAssert(src[simdIndex][0] == data[simdIndex]->get_1D_view_by_index(viewIndex).data()[0]);
    }
    interleave(simdData.get_1D_view_by_index(viewIndex), src, simdElems);
  }

  numViews = simdData.get_num_2D_views();
  for(unsigned viewIndex=0; viewIndex<numViews; ++viewIndex) {
    for(int simdIndex=0; simdIndex<simdElems; ++simdIndex) {
      src[simdIndex] = data[simdIndex]->get_2D_view_by_index(viewIndex).data();
      NGP_ThrowAssert(data[simdIndex]->get_2D_view_by_index(viewIndex).size() == simdData.get_2D_view_by_index(viewIndex).size());
      NGP_ThrowAssert(src[simdIndex] != nullptr);
      NGP_ThrowAssert(src[simdIndex][0] == data[simdIndex]->get_2D_view_by_index(viewIndex).data()[0]);
    }
    interleave(simdData.get_2D_view_by_index(viewIndex), src, simdElems);
  }

  numViews = simdData.get_num_3D_views();
  for(unsigned viewIndex=0; viewIndex<numViews; ++viewIndex) {
    for(int simdIndex=0; simdIndex<simdElems; ++simdIndex) {
      src[simdIndex] = data[simdIndex]->get_3D_view_by_index(viewIndex).data();
      NGP_ThrowAssert(data[simdIndex]->get_3D_view_by_index(viewIndex).size() == simdData.get_3D_view_by_index(viewIndex).size());
      NGP_ThrowAssert(src[simdIndex] != nullptr);
      NGP_ThrowAssert(src[simdIndex][0] == data[simdIndex]->get_3D_view_by_index(viewIndex).data()[0]);
    }
    interleave(simdData.get_3D_view_by_index(viewIndex), src, simdElems);
  }
}

#ifndef KOKKOS_ENABLE_CUDA
inline
void copy_and_interleave(std::unique_ptr<ScratchViews<double>>* data,
                         int simdElems,
                         ScratchViews<DoubleType>& simdData)
{
    MultiDimViews<DoubleType, TeamHandleType, HostShmem>& simdFieldViews = simdData.get_field_views();
    const MultiDimViews<double, TeamHandleType, HostShmem>* fViews[stk::simd::ndoubles] = {nullptr};

    for(int simdIndex=0; simdIndex<simdElems; ++simdIndex) {
      fViews[simdIndex] = &data[simdIndex]->get_field_views();
    }

    copy_and_interleave(fViews, simdElems, simdFieldViews);
}
#endif

KOKKOS_INLINE_FUNCTION
void extract_vector_lane(const SharedMemView<DoubleType*>& simdrhs, int simdIndex, SharedMemView<double*>& rhs)
{
  int dim = simdrhs.extent(0);
  const DoubleType* sr = simdrhs.data();
  double* r = rhs.data();
  for(int i=0; i<dim; ++i) {
    r[i] = stk::simd::get_data(sr[i], simdIndex);
  }
}

KOKKOS_INLINE_FUNCTION
void extract_vector_lane(const SharedMemView<DoubleType**>& simdlhs, int simdIndex, SharedMemView<double**>& lhs)
{
  int len = simdlhs.extent(0)*simdlhs.extent(1);
  const DoubleType* sl = simdlhs.data();
  double* l = lhs.data();
  for(int i=0; i<len; ++i) {
    l[i] = stk::simd::get_data(sl[i], simdIndex);
  }
}

} // namespace nalu
} // namespace Sierra

#endif
