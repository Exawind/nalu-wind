/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MultiDimViews_h
#define MultiDimViews_h

#include <KokkosInterface.h>

namespace sierra {
namespace nalu {

struct NumNeededViews {
  unsigned num1DViews;
  unsigned num2DViews;
  unsigned num3DViews;
  unsigned num4DViews;
};

KOKKOS_FUNCTION
inline
size_t adjust_up_to_alignment_boundary(size_t input, size_t alignment)
{
  size_t remainder = input % alignment;
  if (remainder > 0) {
    return input + alignment - remainder;
  }
  return input;
}

template<typename T, typename TEAMHANDLETYPE=DeviceTeamHandleType, typename SHMEM=DeviceShmem>
class MultiDimViews {
public:
  using SharedMemView1D = SharedMemView<T*,SHMEM>;
  using SharedMemView2D = SharedMemView<T**,SHMEM>;
  using SharedMemView3D = SharedMemView<T***,SHMEM>;
  using SharedMemView4D = SharedMemView<T****,SHMEM>;

  static constexpr unsigned bytesPerUnsigned = sizeof(unsigned);

  KOKKOS_FUNCTION
  MultiDimViews(const TEAMHANDLETYPE& team,
                unsigned maxOrdinal,
                const NumNeededViews& numNeededViews)
  : indices(get_shmem_view_1D<int,TEAMHANDLETYPE,SHMEM>(team,
            adjust_up_to_alignment_boundary((maxOrdinal+1)*bytesPerUnsigned, KOKKOS_MEMORY_ALIGNMENT)/bytesPerUnsigned)),
    views_1D(), views_2D(), views_3D(), views_4D(), 
    views_1D_size(0), views_2D_size(0), views_3D_size(0), views_4D_size(0)
  {
    if (numNeededViews.num1DViews > maxViewsPerDim ||
        numNeededViews.num2DViews > maxViewsPerDim ||
        numNeededViews.num3DViews > maxViewsPerDim ||
        numNeededViews.num4DViews > maxViewsPerDim)
    {
        printf("Too many views per dimension. Each of (%d, %d, %d, %d) must be less than %d. Code will crash...\n",
               numNeededViews.num1DViews, numNeededViews.num2DViews, numNeededViews.num3DViews, numNeededViews.num4DViews,
               maxViewsPerDim);
    }
#ifndef KOKKOS_ENABLE_CUDA
    for(unsigned i=0; i<numNeededViews.num1DViews; ++i) { views_1D[i] = nullptr; }
    for(unsigned i=0; i<numNeededViews.num2DViews; ++i) { views_2D[i] = nullptr; }
    for(unsigned i=0; i<numNeededViews.num3DViews; ++i) { views_3D[i] = nullptr; }
    for(unsigned i=0; i<numNeededViews.num4DViews; ++i) { views_4D[i] = nullptr; }
#else
    for(unsigned i=0; i<numNeededViews.num1DViews; ++i) { new (&views_1D[i]) SharedMemView1D; }
    for(unsigned i=0; i<numNeededViews.num2DViews; ++i) { new (&views_2D[i]) SharedMemView2D; }
    for(unsigned i=0; i<numNeededViews.num3DViews; ++i) { new (&views_3D[i]) SharedMemView3D; }
    for(unsigned i=0; i<numNeededViews.num4DViews; ++i) { new (&views_4D[i]) SharedMemView4D; }
#endif
  }

  KOKKOS_FUNCTION
  ~MultiDimViews()
  {
#ifndef KOKKOS_ENABLE_CUDA
    for(unsigned i=0; i<get_num_1D_views(); ++i) { delete views_1D[i]; }
    for(unsigned i=0; i<get_num_2D_views(); ++i) { delete views_2D[i]; }
    for(unsigned i=0; i<get_num_3D_views(); ++i) { delete views_3D[i]; }
    for(unsigned i=0; i<get_num_4D_views(); ++i) { delete views_4D[i]; }
#endif
  }

  KOKKOS_FUNCTION
  static size_t bytes_needed(unsigned maxOrdinal, const NumNeededViews& numNeededViews)
  {
    return (maxOrdinal+1)*sizeof(unsigned)
          + numNeededViews.num1DViews*sizeof(SharedMemView1D)
          + numNeededViews.num2DViews*sizeof(SharedMemView2D)
          + numNeededViews.num3DViews*sizeof(SharedMemView3D)
          + numNeededViews.num4DViews*sizeof(SharedMemView4D)
          + sizeof(SharedMemView<int*,SHMEM>)
          + sizeof(SharedMemView<SharedMemView1D >)
          + sizeof(SharedMemView<SharedMemView2D >)
          + sizeof(SharedMemView<SharedMemView3D >)
          + sizeof(SharedMemView<SharedMemView4D >);
  }

  KOKKOS_FUNCTION
  SharedMemView1D& get_scratch_view_1D(unsigned ordinal)
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_1D[indices[ordinal]];
#else
    return views_1D[indices[ordinal]];
#endif
  }

  KOKKOS_FUNCTION
  SharedMemView2D& get_scratch_view_2D(unsigned ordinal)
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_2D[indices[ordinal]];
#else
    return views_2D[indices[ordinal]];
#endif
  }

  KOKKOS_FUNCTION
  SharedMemView3D& get_scratch_view_3D(unsigned ordinal)
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_3D[indices[ordinal]];
#else
    return views_3D[indices[ordinal]];
#endif
  }

  KOKKOS_FUNCTION
  SharedMemView4D& get_scratch_view_4D(unsigned ordinal)
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_4D[indices[ordinal]];
#else
    return views_4D[indices[ordinal]];
#endif
  }

  KOKKOS_FUNCTION
  void add_1D_view(unsigned ordinal, const SharedMemView1D& view)
  {
#ifndef KOKKOS_ENABLE_CUDA
    views_1D[views_1D_size] = new SharedMemView1D;
    *views_1D[views_1D_size] = view;
#else
    views_1D[views_1D_size] = view;
#endif
    indices[ordinal] = views_1D_size;
    ++views_1D_size;
  }

  KOKKOS_FUNCTION
  void add_2D_view(unsigned ordinal, const SharedMemView2D& view)
  {
#ifndef KOKKOS_ENABLE_CUDA
    views_2D[views_2D_size] = new SharedMemView2D;
    *views_2D[views_2D_size] = view;
#else
    views_2D[views_2D_size] = view;
#endif
    indices[ordinal] = views_2D_size;
    ++views_2D_size;
  }

  KOKKOS_FUNCTION
  void add_3D_view(unsigned ordinal, const SharedMemView3D& view)
  {
#ifndef KOKKOS_ENABLE_CUDA
    views_3D[views_3D_size] = new SharedMemView3D;
    *views_3D[views_3D_size] = view;
#else
    views_3D[views_3D_size] = view;
#endif
    indices[ordinal] = views_3D_size;
    ++views_3D_size;
  }

  KOKKOS_FUNCTION
  void add_4D_view(unsigned ordinal, const SharedMemView4D& view)
  {
#ifndef KOKKOS_ENABLE_CUDA
    views_4D[views_4D_size] = new SharedMemView4D;
    *views_4D[views_4D_size] = view;
#else
    views_4D[views_4D_size] = view;
#endif
    indices[ordinal] = views_4D_size;
    ++views_4D_size;
  }

  KOKKOS_FUNCTION
  const SharedMemView1D& get_1D_view_by_index(unsigned idx) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_1D[idx];
#else
    return views_1D[idx];
#endif
  }

  KOKKOS_FUNCTION
  const SharedMemView2D& get_2D_view_by_index(unsigned idx) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_2D[idx];
#else
    return views_2D[idx];
#endif
  }

  KOKKOS_FUNCTION
  const SharedMemView3D& get_3D_view_by_index(unsigned idx) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_3D[idx];
#else
    return views_3D[idx];
#endif
  }

  KOKKOS_FUNCTION
  const SharedMemView4D& get_4D_view_by_index(unsigned idx) const
  {
#ifndef KOKKOS_ENABLE_CUDA
    return *views_4D[idx];
#else
    return views_4D[idx];
#endif
  }

  KOKKOS_FUNCTION unsigned get_num_1D_views() const { return views_1D_size; }
  KOKKOS_FUNCTION unsigned get_num_2D_views() const { return views_2D_size; }
  KOKKOS_FUNCTION unsigned get_num_3D_views() const { return views_3D_size; }
  KOKKOS_FUNCTION unsigned get_num_4D_views() const { return views_4D_size; }

public:
  static const unsigned maxViewsPerDim = 25;

  SharedMemView<int*,SHMEM> indices;
#ifndef KOKKOS_ENABLE_CUDA
  SharedMemView1D* views_1D[maxViewsPerDim];
  SharedMemView2D* views_2D[maxViewsPerDim];
  SharedMemView3D* views_3D[maxViewsPerDim];
  SharedMemView4D* views_4D[maxViewsPerDim];
#else
  SharedMemView1D views_1D[maxViewsPerDim];
  SharedMemView2D views_2D[maxViewsPerDim];
  SharedMemView3D views_3D[maxViewsPerDim];
  SharedMemView4D views_4D[maxViewsPerDim];
#endif
  unsigned views_1D_size;
  unsigned views_2D_size;
  unsigned views_3D_size;
  unsigned views_4D_size;
};

}//namespace nalu
}//namespace sierra

#endif

