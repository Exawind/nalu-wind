// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TensorProductSource_h
#define TensorProductSource_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void add_volumetric_source(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order,Scalar>& volume_metric,
  const nodal_scalar_view<poly_order,Scalar>& nodal_source,
  nodal_scalar_view<poly_order,Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        nodal_source(k,j,i) *= volume_metric(k,j,i);
      }
    }
  }
  ops.volume(nodal_source, rhs);
}

template <int poly_order, typename Scalar>
void add_volumetric_source(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order,Scalar>& volume_metric,
  const nodal_vector_view<poly_order,Scalar>& nodal_source,
  nodal_vector_view<poly_order,Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        nodal_source(k,j,i, XH) *= volume_metric(k,j,i);
        nodal_source(k,j,i, YH) *= volume_metric(k,j,i);
        nodal_source(k,j,i, ZH) *= volume_metric(k,j,i);
      }
    }
  }
  ops.volume(nodal_source, rhs);
}

template <int poly_order, typename Scalar, typename SourceFunc>
void add_volumetric_source_func(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& volume_metric,
  nodal_scalar_view<poly_order, Scalar>& rhs,
  SourceFunc func)
{
  nodal_scalar_workview<poly_order,Scalar> l_source(0);
  auto& source = l_source.view();

  for (int k = 0; k < poly_order + 1; ++k) {
    for (int j = 0; j < poly_order + 1; ++j) {
      for (int i = 0; i < poly_order + 1; ++i) {
        source(k,j,i) = func(k,j,i) * volume_metric(k,j,i);
      }
    }
  }
  ops.volume(source, rhs);
}


}
}
}

#endif
