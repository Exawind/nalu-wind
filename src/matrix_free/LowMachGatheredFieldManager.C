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
#include "matrix_free/ElementSCSInterpolate.h"

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
stk::mesh::NgpField<T>&
get_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

template <int p>
LowMachGatheredFieldManager<p>::LowMachGatheredFieldManager(
  stk::mesh::BulkData& bulk_in, stk::mesh::Selector active_in)
  : bulk(bulk_in),
    meta(bulk_in.mesh_meta_data()),
    active(active_in),
    conn(stk_connectivity_map<p>(bulk.get_updated_ngp_mesh(), active)),
    scratch_volume_metric("scratch_volume_metric", conn.extent_int(0))
{
}

template <int p>
void
LowMachGatheredFieldManager<p>::gather_all()
{
  stk::mesh::ProfilingBlock pf("LowMachGatheredFieldManager<p>::gather_all");
  fields = gather_required_lowmach_fields<p>(meta, conn);

  Kokkos::deep_copy(scratch_volume_metric, fields.volume_metric);
  coefficient_fields.unscaled_volume_metric = fields.unscaled_volume_metric;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;
  coefficient_fields.laplacian_metric = fields.laplacian_metric;
  coefficient_fields.advection_metric = fields.advection_metric;
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
  auto vel = get_ngp_field<double>(
    meta, lowmach_info::velocity_name, stk::mesh::StateNP1);
  field_gather<p>(conn, vel, fields.up1);
}

template <int p>
void
LowMachGatheredFieldManager<p>::update_pressure()
{
  auto pressure = get_ngp_field<double>(meta, lowmach_info::pressure_name);
  field_gather<p>(conn, pressure, fields.pressure);
}
template <int p>
void
LowMachGatheredFieldManager<p>::update_grad_p()
{
  auto gp = get_ngp_field<double>(meta, lowmach_info::pressure_grad_name);
  field_gather<p>(conn, gp, fields.gp);
}

namespace {

template <
  int p,
  int dir,
  typename ViscOldArray,
  typename ViscNewArray,
  typename MetricArray>
KOKKOS_FORCEINLINE_FUNCTION void
rescale_metric(
  const ViscOldArray& old_visc,
  const ViscNewArray& new_visc,
  MetricArray& metric)
{
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        const auto visc_ratio = interp_scs<p, dir>(new_visc, l, s, r) /
                                interp_scs<p, dir>(old_visc, l, s, r);
        for (int di = 0; di < 3; ++di) {
          metric(dir, l, s, r, di) *= visc_ratio;
        }
      }
    }
  }
}

template <int p>
void
update_transport_coefficients_impl(
  const const_elem_mesh_index_view<p>& conn,
  const stk::mesh::NgpField<double>& rho_field,
  const stk::mesh::NgpField<double>& visc_field,
  scalar_view<p> rho,
  scalar_view<p> visc,
  scalar_view<p> vp1,
  scs_vector_view<p> diff)
{
  Kokkos::parallel_for(
    conn.extent_int(0), KOKKOS_LAMBDA(int index) {
      const auto length = valid_offset<p>(index, conn);
      LocalArray<ftype[p + 1][p + 1][p + 1]> visc_tmp;
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < length; ++n) {
              const auto mesh_index = conn(index, k, j, i, n);
              const auto rescaled_vol =
                stk::simd::get_data(vp1(index, k, j, i), n) *
                rho_field.get(mesh_index, 0) /
                stk::simd::get_data(rho(index, k, j, i), n);

              stk::simd::set_data(vp1(index, k, j, i), n, rescaled_vol);
              stk::simd::set_data(
                rho(index, k, j, i), n, rho_field.get(mesh_index, 0));
              stk::simd::set_data(
                visc_tmp(k, j, i), n, visc_field.get(mesh_index, 0));
            }
          }
        }
      }
      auto visc_v =
        Kokkos::subview(visc, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto diff_v = Kokkos::subview(
        diff, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL);
      rescale_metric<p, 0>(visc_v, visc_tmp, diff_v);
      rescale_metric<p, 1>(visc_v, visc_tmp, diff_v);
      rescale_metric<p, 2>(visc_v, visc_tmp, diff_v);

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            visc_v(k, j, i) = visc_tmp(k, j, i);
          }
        }
      }
    });
}

} // namespace

template <int p>
void
LowMachGatheredFieldManager<p>::update_transport_coefficients()
{
  stk::mesh::ProfilingBlock pf(
    "LowMachGatheredFieldManager<p>::update_transport_coefficients");

  auto rho_field = get_ngp_field(meta, lowmach_info::density_name);
  rho_field.sync_to_device();

  auto visc_field = get_ngp_field(meta, lowmach_info::viscosity_name);
  visc_field.sync_to_device();

  update_transport_coefficients_impl<p>(
    conn, rho_field, visc_field, fields.rho, fields.mu, fields.volume_metric,
    fields.diffusion_metric);
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
