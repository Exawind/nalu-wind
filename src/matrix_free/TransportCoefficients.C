// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/TransportCoefficients.h"

#include "matrix_free/ElementGradient.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/LowMachInfo.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ShuffledAccess.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/GeometricFunctions.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_simd/Simd.hpp"
#include "stk_util/util/ReportHandler.hpp"

#include <iosfwd>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

KOKKOS_FUNCTION ftype
wale_gradient_invariant(const ArrayND<ftype[3][3]>& dudx)
{
  const auto dudx_sq = square(dudx);
  const auto one_third_trace =
    (1. / 3.) * (dudx_sq(0, 0) + dudx_sq(1, 1) + dudx_sq(2, 2));

  ftype sij_sq = 0;
  ftype sijd_sq = 0;
  for (int dj = 0; dj < 3; ++dj) {
    for (int di = 0; di < 3; ++di) {
      const auto sij = 0.5 * (dudx(dj, di) + dudx(di, dj));
      const auto trace_kron = (dj == di) * one_third_trace;
      const auto sijd = 0.5 * (dudx_sq(dj, di) + dudx_sq(di, dj)) - trace_kron;
      sij_sq += sij * sij;
      sijd_sq += sijd * sijd;
    }
  }
  constexpr double small = 1.e-8;
  const auto num = stk::math::pow(sijd_sq, 1.5) + small * small;
  const auto den =
    stk::math::pow(sij_sq, 2.5) + stk::math::pow(sijd_sq, 1.25) + small;

  return num / den;
}

KOKKOS_FUNCTION ftype
smag_gradient_invariant(const ArrayND<ftype[3][3]>& dudx)
{
  ftype sij_sq = 0;
  for (int dj = 0; dj < 3; ++dj) {
    for (int di = 0; di < 3; ++di) {
      const auto sij = 0.5 * (dudx(dj, di) + dudx(di, dj));
      sij_sq += sij * sij;
    }
  }
  return stk::math::sqrt(2 * sij_sq);
}

template <
  int p,
  int dir,
  typename ViscArray,
  typename ConstMetricArray,
  typename MetricArray>
KOKKOS_FORCEINLINE_FUNCTION void
scale_metric(
  const ViscArray& visc,
  const ConstMetricArray& unscaled_metric,
  MetricArray& metric)
{
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        const auto edge_visc =
          stk::math::max(0, interp_scs<p, dir>(visc, l, s, r));
        for (int di = 0; di < 3; ++di) {
          metric(dir, l, s, r, di) =
            unscaled_metric(dir, l, s, r, di) * edge_visc;
        }
      }
    }
  }
}

} // namespace

namespace impl {
template <int p>
void
transport_coefficients_t<p>::invoke(
  GradTurbModel model,
  const const_elem_mesh_index_view<p>& conn,
  const stk::mesh::NgpField<double>& rho_f,
  const stk::mesh::NgpField<double>& mu_f,
  const_scalar_view<p> filter_scale,
  const_vector_view<p> xc,
  const_vector_view<p> vel,
  const_scalar_view<p> unscaled_vol,
  const_scs_vector_view<p> unscaled_diff,
  scalar_view<p> rho,
  scalar_view<p> visc,
  scalar_view<p> vol,
  scs_vector_view<p> diff)
{
  Kokkos::parallel_for(
    DeviceRangePolicy(0, conn.extent_int(0)), KOKKOS_LAMBDA(int index) {
      {
        const auto box = hex_vertex_coordinates<p>(index, xc);
        auto uvec = Kokkos::subview(
          vel, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        for (int k = 0; k < p + 1; ++k) {
          for (int j = 0; j < p + 1; ++j) {
            for (int i = 0; i < p + 1; ++i) {
              ftype node_lam_visc = 0;
              ftype node_rho = 0;
              for (int n = 0; n < simd_len; ++n) {
                const auto mesh_index =
                  MeshIndexGetter<p, simd_len>::get(conn, index, k, j, i, n);
                stk::simd::set_data(node_rho, n, rho_f.get(mesh_index, 0));
                stk::simd::set_data(node_lam_visc, n, mu_f.get(mesh_index, 0));
              }

              vol(index, k, j, i) = node_rho * unscaled_vol(index, k, j, i);

              const auto dudx = gradient_nodal<p>(box, uvec, k, j, i);
              ftype invariant;
              switch (model) {
              case GradTurbModel::WALE: {
                invariant = wale_gradient_invariant(dudx);
                break;
              }
              case GradTurbModel::SMAG: {
                invariant = smag_gradient_invariant(dudx);
                break;
              }
              default: {
                invariant = 0;
              }
              }
              const auto ls = filter_scale(index, k, j, i);
              const auto tvisc = node_rho * ls * ls * invariant;
              visc(index, k, j, i) = node_lam_visc + tvisc;
              rho(index, k, j, i) = node_rho;
            }
          }
        }
      }
      auto visc_v =
        Kokkos::subview(visc, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      auto unscaled_diff_v = Kokkos::subview(
        unscaled_diff, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL, Kokkos::ALL);
      auto diff_v = Kokkos::subview(
        diff, index, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
        Kokkos::ALL);
      scale_metric<p, 0>(visc_v, unscaled_diff_v, diff_v);
      scale_metric<p, 1>(visc_v, unscaled_diff_v, diff_v);
      scale_metric<p, 2>(visc_v, unscaled_diff_v, diff_v);
    });
}
INSTANTIATE_POLYSTRUCT(transport_coefficients_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
