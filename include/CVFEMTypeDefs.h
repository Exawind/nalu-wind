/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CVFEMTypeDefs_h
#define CVFEMTypeDefs_h

#include <ScratchWorkView.h>
#include <SimdInterface.h>
#include <KokkosInterface.h>

#include <array>

namespace sierra {
namespace nalu {

template <typename ArrayType> using ViewType = Kokkos::View<ArrayType,
    Kokkos::LayoutRight, HostShmem, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned>>;

template <int p, typename Scalar = DoubleType>
struct CVFEMViews
{
  static constexpr int n1D = p + 1;
  static constexpr int nscs = p;
  static constexpr int dim = 3;
  static constexpr int npe = n1D*n1D*n1D;

  using nodal_scalar_array = Scalar[n1D][n1D][n1D];
  using nodal_scalar_view = ViewType<nodal_scalar_array>;
  using nodal_scalar_wsv = ScratchWorkView<npe, nodal_scalar_view>;
  //--------------------------------------------------------------------------
  using nodal_vector_array = Scalar[n1D][n1D][n1D][dim];
  using nodal_vector_view = ViewType<nodal_vector_array>;
  using nodal_vector_wsv = ScratchWorkView<dim*n1D*n1D*n1D, nodal_vector_view>;
  //--------------------------------------------------------------------------
  using nodal_tensor_array = Scalar[n1D][n1D][n1D][dim][dim];
  using nodal_tensor_view = ViewType<nodal_tensor_array>;
  using nodal_tensor_wsv = ScratchWorkView<dim*dim*n1D*n1D*n1D, nodal_tensor_view>;
  //--------------------------------------------------------------------------
  using scs_scalar_array = Scalar[dim][n1D][n1D][n1D];
  using scs_scalar_view = ViewType<scs_scalar_array>;
  using scs_scalar_wsv = ScratchWorkView<dim*n1D*n1D*n1D, scs_scalar_view>;
  //--------------------------------------------------------------------------
  using scs_vector_array = Scalar[dim][n1D][n1D][n1D][dim];
  using scs_vector_view = ViewType<scs_vector_array>;
  using scs_vector_wsv = ScratchWorkView<dim*dim*n1D*n1D*n1D, scs_vector_view>;
  //--------------------------------------------------------------------------
  using scs_symmtensor_array = Scalar[dim][n1D][n1D][n1D][dim][dim];
  using scs_symmtensor_view = ViewType<scs_symmtensor_array>;
  using scs_symmtensor_wsv = ScratchWorkView<dim*dim*dim*n1D*n1D*n1D, scs_symmtensor_view>;
  //--------------------------------------------------------------------------
  using scs_tensor_array = Scalar[dim][n1D][n1D][n1D][dim][dim];
  using scs_tensor_view = ViewType<scs_tensor_array>;
  using scs_tensor_wsv = ScratchWorkView<dim*dim*dim*n1D*n1D*n1D, scs_tensor_view>;
  //--------------------------------------------------------------------------
  using matrix_array = Scalar[npe][npe];
  using matrix_view = ViewType<matrix_array>;
  //--------------------------------------------------------------------------
  using matrix_vector_array = Scalar[dim*npe][dim*npe];
  using matrix_vector_view = ViewType<matrix_vector_array>;
};

template <int p, typename Scalar = DoubleType>
using nodal_scalar_view = typename CVFEMViews<p,Scalar>::nodal_scalar_view;

template <int p, typename Scalar = DoubleType>
using nodal_scalar_workview = typename CVFEMViews<p,Scalar>::nodal_scalar_wsv;

template <int p, typename Scalar = DoubleType>
using nodal_vector_view = typename CVFEMViews<p,Scalar>::nodal_vector_view;

template <int p, typename Scalar = DoubleType>
using nodal_vector_workview = typename CVFEMViews<p,Scalar>::nodal_vector_wsv;

template <int p, typename Scalar = DoubleType>
using nodal_tensor_view = typename CVFEMViews<p,Scalar>::nodal_tensor_view;

template <int p, typename Scalar = DoubleType>
using nodal_tensor_workview = typename CVFEMViews<p,Scalar>::nodal_tensor_wsv;

template <int p, typename Scalar = DoubleType>
using scs_scalar_view = typename CVFEMViews<p,Scalar>::scs_scalar_view;

template <int p, typename Scalar = DoubleType>
using scs_scalar_workview = typename CVFEMViews<p,Scalar>::scs_scalar_wsv;

template <int p, typename Scalar = DoubleType>
using scs_vector_view = typename CVFEMViews<p,Scalar>::scs_vector_view;

template <int p, typename Scalar = DoubleType>
using scs_vector_workview = typename CVFEMViews<p,Scalar>::scs_vector_wsv;

template <int p, typename Scalar = DoubleType>
using scs_symmtensor_view = typename CVFEMViews<p,Scalar>::scs_symmtensor_view;

template <int p, typename Scalar = DoubleType>
using scs_symmtensor_workview = typename CVFEMViews<p,Scalar>::scs_symmtensor_wsv;

template <int p, typename Scalar = DoubleType>
using scs_tensor_view = typename CVFEMViews<p,Scalar>::scs_tensor_view;

template <int p, typename Scalar = DoubleType>
using scs_tensor_workview = typename CVFEMViews<p,Scalar>::scs_tensor_wsv;

template <int p, typename Scalar = DoubleType>
using matrix_view = typename CVFEMViews<p, Scalar>::matrix_view;

template <int p, typename Scalar = DoubleType>
using matrix_vector_view = typename CVFEMViews<p, Scalar>::matrix_vector_view;

template<int p, typename Scalar = DoubleType>
using HexViews = CVFEMViews<p, Scalar>;

template <typename ArrayType>
using CoeffViewType = Kokkos::View<ArrayType, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Aligned>>;

template <int p, typename Scalar = DoubleType>
using nodal_matrix_array = Scalar[p+1][p+1];

template <int p, typename Scalar = DoubleType>
using nodal_matrix_view = CoeffViewType<nodal_matrix_array<p,Scalar>>;
//--------------------------------------------------------------------------
template <int p, typename Scalar = DoubleType>
using scs_matrix_array = Scalar[p+1][p+1]; // always pad to be square

template <int p, typename Scalar = DoubleType>
using scs_matrix_view = CoeffViewType<scs_matrix_array<p,Scalar>>;
//--------------------------------------------------------------------------
template <int p, typename Scalar = DoubleType>
using linear_nodal_matrix_array = Scalar[2][p+1];

template <int p, typename Scalar = DoubleType>
using linear_nodal_matrix_view = CoeffViewType<linear_nodal_matrix_array<p,Scalar>>;
//--------------------------------------------------------------------------
template <int p, typename Scalar = DoubleType>
using linear_scs_matrix_array = Scalar[2][p];

template <int p, typename Scalar = DoubleType>
using linear_scs_matrix_view = CoeffViewType<linear_scs_matrix_array<p,Scalar>>;
//--------------------------------------------------------------------------
using node_map_view = Kokkos::View<int*>;

#define CVFEMTypeDefsDim(x,y,z) \
  using y##_##z##_##view = typename x::y##_##z##_##view; \
  using y##_##z##_##workview = typename x::y##_##z##_##wsv;

#define DeclareCVFEMTypeDefs(x) \
  CVFEMTypeDefsDim(x,nodal,scalar) \
  CVFEMTypeDefsDim(x,nodal,vector) \
  CVFEMTypeDefsDim(x,nodal,tensor) \
  CVFEMTypeDefsDim(x,scs,scalar) \
  CVFEMTypeDefsDim(x,scs,vector) \
  CVFEMTypeDefsDim(x,scs,symmtensor) \
  CVFEMTypeDefsDim(x,scs,tensor) \
  using matrix##_##view = typename x::matrix##_##view; \
  using matrix##_##vector##_##view = typename x::matrix##_##vector##_##view

} // namespace nalu
} // namespace Sierra

#endif
