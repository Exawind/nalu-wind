// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LowMachGatheredFieldManager.h"
#include "matrix_free/LowMachInfo.h"

#include "matrix_free/LowMachFields.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/TransportCoefficients.h"

#include "Kokkos_Macros.hpp"

#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"

#include <iosfwd>
#include <stk_simd/Simd.hpp>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename T = double>
stk::mesh::NgpField<T>
get_synced_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));

  auto field = stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  field.sync_to_device();
  return field;
}

template <int p>
LowMachGatheredFieldManager<p>::LowMachGatheredFieldManager(
  stk::mesh::BulkData& bulk_in, stk::mesh::Selector active_in)
  : bulk(bulk_in),
    meta(bulk_in.mesh_meta_data()),
    active(active_in),
    conn(stk_connectivity_map<p>(bulk.get_updated_ngp_mesh(), active)),
    filter_scale("scaled filter length", conn.extent_int(0))
{
}

template <int p>
void
LowMachGatheredFieldManager<p>::gather_all()
{
  stk::mesh::ProfilingBlock pf("LowMachGatheredFieldManager<p>::gather_all");
  fields = gather_required_lowmach_fields<p>(meta, conn);
  coefficient_fields.unscaled_volume_metric = fields.unscaled_volume_metric;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;
  coefficient_fields.laplacian_metric = fields.laplacian_metric;
  coefficient_fields.advection_metric = fields.advection_metric;

  auto filter_scale_field =
    get_synced_ngp_field(meta, info::scaled_filter_length_name);
  field_gather<p>(conn, filter_scale_field, filter_scale);
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_fields()
{
  update_velocity();
  update_pressure();
  update_grad_p();
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_velocity()
{
  auto vel = get_synced_ngp_field<double>(
    meta, info::velocity_name, stk::mesh::StateNP1);
  field_gather<p>(conn, vel, fields.up1);
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_pressure()
{
  auto pressure = get_synced_ngp_field<double>(meta, info::pressure_name);
  field_gather<p>(conn, pressure, fields.pressure);
}
template <int p>
void
LowMachGatheredFieldManager<p>::update_grad_p()
{
  auto gp = get_synced_ngp_field<double>(meta, info::pressure_grad_name);
  field_gather<p>(conn, gp, fields.gp);
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_transport_coefficients(
  GradTurbModel model)
{
  stk::mesh::ProfilingBlock pf(
    "LowMachGatheredFieldManager<p>::update_transport_coefficients");

  auto rho_field = get_synced_ngp_field(meta, info::density_name);
  auto visc_field = get_synced_ngp_field(meta, info::viscosity_name);

  transport_coefficients<p>(
    model, conn, rho_field, visc_field, filter_scale, fields.xc, fields.up1,
    fields.unscaled_volume_metric, fields.laplacian_metric, fields.rho,
    fields.mu, fields.volume_metric, fields.diffusion_metric);
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_mdot(double scaling)
{
  geom::linear_advection_metric<p>(
    scaling, fields.area_metric, fields.laplacian_metric, fields.rho,
    fields.up1, fields.gp, fields.pressure, fields.advection_metric);
}

template <int p>
void
LowMachGatheredFieldManager<p>::swap_states()
{
  auto um1 = fields.um1;
  fields.um1 = fields.up0;
  fields.up0 = fields.up1;
  fields.up1 = um1;

  auto vm1 = fields.vm1;
  fields.vm1 = fields.vp0;
  fields.vp0 = fields.volume_metric;
  fields.volume_metric = vm1;
}
INSTANTIATE_POLYCLASS(LowMachGatheredFieldManager);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
