// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOWMACH_FIELDS_H
#define LOWMACH_FIELDS_H

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"

namespace stk {
namespace mesh {
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
struct LowMachResidualFields
{
  // fields
  vector_view<p> xc;
  scalar_view<p> rho;
  scalar_view<p> mu;
  vector_view<p> up0;
  vector_view<p> up1;
  vector_view<p> um1;
  scalar_view<p> pressure;
  vector_view<p> gp;
  vector_view<p> force;

  // geometric terms
  scalar_view<p> unscaled_volume_metric;
  scalar_view<p> vm1;
  scalar_view<p> vp0;
  scalar_view<p> volume_metric;
  scs_scalar_view<p> advection_metric;
  scs_vector_view<p> area_metric;
  scs_vector_view<p> diffusion_metric;
  scs_vector_view<p> laplacian_metric;
};

template <int p>
struct LowMachLinearizedResidualFields
{
  scalar_view<p> unscaled_volume_metric;
  scalar_view<p> volume_metric;
  scs_scalar_view<p> advection_metric;
  scs_vector_view<p> diffusion_metric;
  scs_vector_view<p> laplacian_metric;
};

template <int p>
struct LowMachBCFields
{
  node_vector_view up1;
  node_vector_view ubc;
  face_scalar_view<p> exposed_pressure;
  face_vector_view<p> exposed_areas;
};

namespace impl {

template <int p>
struct gather_required_lowmach_fields_t
{
  static LowMachResidualFields<p>
  invoke(const stk::mesh::MetaData&, const_elem_mesh_index_view<p>);
};

} // namespace impl
P_INVOKEABLE(gather_required_lowmach_fields)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
