// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LowMachFields.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/LowMachInfo.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/KokkosViewTypes.h"

#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <typename T = double>
stk::mesh::NgpField<T>
get_and_sync_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  STK_ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  STK_ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));

  auto field = stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  field.sync_to_device();
  return field;
}

template <int p>
LowMachResidualFields<p>
gather_required_lowmach_fields_t<p>::invoke(
  const stk::mesh::MetaData& meta, const_elem_mesh_index_view<p> conn)
{
  LowMachResidualFields<p> fields;

  fields.up1 = vector_view<p>{"up1", conn.extent(0)};
  field_gather<p>(
    conn,
    get_and_sync_ngp_field(
      meta, lowmach_info::velocity_name, stk::mesh::StateNP1),
    fields.up1);
  fields.up0 = vector_view<p>{"up0", conn.extent(0)};
  field_gather<p>(
    conn,
    get_and_sync_ngp_field(
      meta, lowmach_info::velocity_name, stk::mesh::StateN),
    fields.up0);
  fields.um1 = vector_view<p>{"um1", conn.extent(0)};
  field_gather<p>(
    conn,
    get_and_sync_ngp_field(
      meta, lowmach_info::velocity_name, stk::mesh::StateNM1),
    fields.um1);

  fields.gp = vector_view<p>{"gp", conn.extent(0)};
  field_gather<p>(
    conn, get_and_sync_ngp_field(meta, lowmach_info::pressure_grad_name),
    fields.gp);

  fields.force = vector_view<p>{"force", conn.extent(0)};
  field_gather<p>(
    conn, get_and_sync_ngp_field(meta, lowmach_info::force_name), fields.force);

  fields.pressure = scalar_view<p>{"pressure", conn.extent(0)};
  field_gather<p>(
    conn, get_and_sync_ngp_field(meta, lowmach_info::pressure_grad_name),
    fields.pressure);

  fields.xc = vector_view<p>{"coords", conn.extent(0)};
  field_gather<p>(
    conn, get_and_sync_ngp_field(meta, lowmach_info::coord_name), fields.xc);

  fields.rho = scalar_view<p>{"rp1", conn.extent(0)};
  field_gather<p>(
    conn,
    get_and_sync_ngp_field(
      meta, lowmach_info::density_name, stk::mesh::StateNP1),
    fields.rho);

  fields.mu = scalar_view<p>{"mu", conn.extent(0)};
  field_gather<p>(
    conn, get_and_sync_ngp_field(meta, lowmach_info::viscosity_name),
    fields.mu);

  fields.vm1 = geom::volume_metric<p>(fields.rho, fields.xc);
  fields.vp0 = geom::volume_metric<p>(fields.rho, fields.xc);
  fields.volume_metric = geom::volume_metric<p>(fields.rho, fields.xc);
  fields.unscaled_volume_metric = geom::volume_metric<p>(fields.xc);
  fields.diffusion_metric = geom::diffusion_metric<p>(fields.mu, fields.xc);
  fields.laplacian_metric = geom::diffusion_metric<p>(fields.xc);
  fields.area_metric = geom::linear_areas<p>(fields.xc);

  fields.advection_metric = scs_scalar_view<p>{"mdot", conn.extent(0)};
  geom::linear_advection_metric<p>(
    0., fields.area_metric, fields.laplacian_metric, fields.rho, fields.up1,
    fields.gp, fields.pressure, fields.advection_metric);

  return fields;
}
INSTANTIATE_POLYSTRUCT(gather_required_lowmach_fields_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
