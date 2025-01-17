// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <cmath>
#include <random>

#include "matrix_free/TransportCoefficients.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/StkLowMachFixture.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/TensorOperations.h"

#include "StkSimdComparisons.h"
#include "gtest/gtest.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

class TransportCoefficientsFixture : public LowMachFixture
{
public:
  TransportCoefficientsFixture()
    : LowMachFixture(2, 1),
      conn_(stk_connectivity_map<order>(mesh(), active())),
      filter_scale_("scaled_filter_length", conn_.extent(0))
  {
    Kokkos::deep_copy(filter_scale_, 1);
    skew_mesh();
  }

  void skew_mesh()
  {
    ArrayND<double[3][3]> Q = {{{2, 1, 1.3333}, {0, 2, -1}, {1, 0, 2}}};
    auto coords = stk::mesh::get_updated_ngp_field<double>(coordinate_field());

    stk::mesh::for_each_entity_run(
      mesh(), stk::topology::NODE_RANK, active(),
      KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
        Kokkos::Array<double, 3> xo = {
          {coords.get(mi, 0), coords.get(mi, 1), coords.get(mi, 2)}};
        Kokkos::Array<double, 3> xh;
        transform(Q, xo, xh);
        coords.get(mi, 0) = 1 + xh[0];
        coords.get(mi, 1) = 3 + xh[1];
        coords.get(mi, 2) = -4 + xh[2];
      });
    coords.modify_on_device();
  }

  void linear_ic(double scaling)
  {
    auto vel = stk::mesh::get_updated_ngp_field<double>(velocity_field);
    auto coords = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
    stk::mesh::for_each_entity_run(
      mesh(), stk::topology::NODE_RANK, active(),
      KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
        vel.get(mi, 0) = scaling * coords.get(mi, 1);
        vel.get(mi, 1) = 0;
        vel.get(mi, 2) = 0;
      });
  }

  void plane_strain()
  {
    auto vel = stk::mesh::get_updated_ngp_field<double>(velocity_field);
    auto coords = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
    stk::mesh::for_each_entity_run(
      mesh(), stk::topology::NODE_RANK, active(),
      KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
        vel.get(mi, 0) = coords.get(mi, 1);
        vel.get(mi, 1) = coords.get(mi, 0);
        vel.get(mi, 2) = 0;
      });
  }

  void update_transport_coefficients(
    GradTurbModel model, LowMachResidualFields<order> fields)
  {
    transport_coefficients<order>(
      model, conn_, stk::mesh::get_updated_ngp_field<double>(density_field),
      stk::mesh::get_updated_ngp_field<double>(viscosity_field), filter_scale_,
      fields.xc, fields.up1, fields.unscaled_volume_metric,
      fields.laplacian_metric, fields.rho, fields.mu, fields.volume_metric,
      fields.diffusion_metric);
  }
  const elem_mesh_index_view<order> conn_;
  const scalar_view<order> filter_scale_;
};

TEST_F(
  TransportCoefficientsFixture,
  viscous_layer_does_not_produce_wale_turbulent_viscosity)
{
  auto visc = stk::mesh::get_updated_ngp_field<double>(viscosity_field);
  visc.sync_to_device();
  auto vel = stk::mesh::get_updated_ngp_field<double>(velocity_field);
  vel.sync_to_device();
  auto coords = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  coords.sync_to_device();

  const double some_number = 2.2;
  visc.set_all(mesh(), some_number);

  linear_ic(1.);

  auto fields = gather_required_lowmach_fields<order>(meta, conn_);
  update_transport_coefficients(GradTurbModel::WALE, fields);

  auto laplace_h = Kokkos::create_mirror_view(fields.laplacian_metric);
  Kokkos::deep_copy(laplace_h, fields.laplacian_metric);

  auto diff_h = Kokkos::create_mirror_view(fields.diffusion_metric);
  Kokkos::deep_copy(diff_h, fields.diffusion_metric);

  const int index = 0;
  for (int dj = 0; dj < 3; ++dj) {
    for (int s = 0; s < order + 1; ++s) {
      for (int r = 0; r < order + 1; ++r) {
        for (int d = 0; d < 3; ++d) {
          EXPECT_DOUBLETYPE_NEAR(
            some_number * laplace_h(index, dj, order - 1, s, r, d),
            diff_h(index, dj, order - 1, s, r, d), 1.0e-10);
        }
      }
    }
  }
}

TEST_F(
  TransportCoefficientsFixture, planar_strain_produces_wale_turbulent_viscosity)
{
  const double some_number = 2.2;
  auto visc = stk::mesh::get_updated_ngp_field<double>(viscosity_field);
  visc.set_all(mesh(), some_number);

  plane_strain();

  auto fields = gather_required_lowmach_fields<order>(meta, conn_);
  update_transport_coefficients(GradTurbModel::WALE, fields);

  auto laplace_h = Kokkos::create_mirror_view(fields.laplacian_metric);
  Kokkos::deep_copy(laplace_h, fields.laplacian_metric);

  auto diff_h = Kokkos::create_mirror_view(fields.diffusion_metric);
  Kokkos::deep_copy(diff_h, fields.diffusion_metric);

  const int index = 0;
  ftype max_diff = -1;
  for (int dj = 0; dj < 3; ++dj) {
    for (int s = 0; s < order + 1; ++s) {
      for (int r = 0; r < order + 1; ++r) {
        for (int d = 0; d < 3; ++d) {
          max_diff = stk::math::max(
            stk::math::abs(
              some_number * laplace_h(index, dj, order - 1, s, r, d) -
              diff_h(index, dj, order - 1, s, r, d)),
            max_diff);
        }
      }
    }
  }
  EXPECT_GT(stk::simd::get_data(max_diff, 0), 1e-10);
}

TEST_F(TransportCoefficientsFixture, one_component_linear_smagorinsky_is_exact)
{
  const double lam_visc = 2.2;
  auto visc = stk::mesh::get_updated_ngp_field<double>(viscosity_field);
  visc.set_all(mesh(), lam_visc);

  const double smag_visc = 4;
  linear_ic(smag_visc);

  auto fields = gather_required_lowmach_fields<order>(meta, conn_);
  update_transport_coefficients(GradTurbModel::SMAG, fields);

  auto laplace_h = Kokkos::create_mirror_view(fields.laplacian_metric);
  Kokkos::deep_copy(laplace_h, fields.laplacian_metric);

  auto diff_h = Kokkos::create_mirror_view(fields.diffusion_metric);
  Kokkos::deep_copy(diff_h, fields.diffusion_metric);

  const int index = 0;
  for (int dj = 0; dj < 3; ++dj) {
    for (int s = 0; s < order + 1; ++s) {
      for (int r = 0; r < order + 1; ++r) {
        for (int d = 0; d < 3; ++d) {
          EXPECT_DOUBLETYPE_NEAR(
            (lam_visc + smag_visc) * laplace_h(index, dj, order - 1, s, r, d),
            diff_h(index, dj, order - 1, s, r, d), 1.0e-10);
        }
      }
    }
  }
}

TEST_F(TransportCoefficientsFixture, coefficients_are_updated)
{
  const double some_number_mu = 2.2;
  const double some_number_rho = 9.7;

  auto mu = stk::mesh::get_updated_ngp_field<double>(viscosity_field);
  auto rho = stk::mesh::get_updated_ngp_field<double>(density_field);
  mu.set_all(mesh(), some_number_mu);
  rho.set_all(mesh(), some_number_rho);

  auto fields = gather_required_lowmach_fields<order>(meta, conn_);
  update_transport_coefficients(GradTurbModel::LAM, fields);

  auto laplace_h = Kokkos::create_mirror_view(fields.laplacian_metric);
  Kokkos::deep_copy(laplace_h, fields.laplacian_metric);

  auto diff_h = Kokkos::create_mirror_view(fields.diffusion_metric);
  Kokkos::deep_copy(diff_h, fields.diffusion_metric);

  for (int dj = 0; dj < 3; ++dj) {
    for (int s = 0; s < order + 1; ++s) {
      for (int r = 0; r < order + 1; ++r) {
        for (int d = 0; d < 3; ++d) {
          EXPECT_DOUBLETYPE_NEAR(
            some_number_mu * laplace_h(0, dj, 0, s, r, d),
            diff_h(0, dj, 0, s, r, d), 1.0e-10);
        }
      }
    }
  }

  auto vol_h = Kokkos::create_mirror_view(fields.unscaled_volume_metric);
  Kokkos::deep_copy(vol_h, fields.unscaled_volume_metric);

  auto scaled_vol_h = Kokkos::create_mirror_view(fields.volume_metric);
  Kokkos::deep_copy(scaled_vol_h, fields.volume_metric);

  for (int k = 0; k < order + 1; ++k) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        EXPECT_DOUBLETYPE_NEAR(
          some_number_rho * vol_h(0, k, j, i), scaled_vol_h(0, k, j, i),
          1.0e-8);
      }
    }
  }

  const double some_other_number_mu = 1.4;
  const double some_other_number_rho = 4.12;

  mu.set_all(mesh(), some_other_number_mu);
  rho.set_all(mesh(), some_other_number_rho);

  update_transport_coefficients(GradTurbModel::LAM, fields);

  Kokkos::deep_copy(laplace_h, fields.laplacian_metric);
  Kokkos::deep_copy(diff_h, fields.diffusion_metric);

  for (int dj = 0; dj < 3; ++dj) {
    for (int s = 0; s < order + 1; ++s) {
      for (int r = 0; r < order + 1; ++r) {
        for (int d = 0; d < 3; ++d) {
          EXPECT_DOUBLETYPE_NEAR(
            some_other_number_mu * laplace_h(0, dj, 0, s, r, d),
            diff_h(0, dj, 0, s, r, d), 1.0e-8);
        }
      }
    }
  }

  Kokkos::deep_copy(vol_h, fields.unscaled_volume_metric);
  Kokkos::deep_copy(scaled_vol_h, fields.volume_metric);

  for (int k = 0; k < order + 1; ++k) {
    for (int j = 0; j < order + 1; ++j) {
      for (int i = 0; i < order + 1; ++i) {
        EXPECT_DOUBLETYPE_NEAR(
          some_other_number_rho * vol_h(0, k, j, i), scaled_vol_h(0, k, j, i),
          1.0e-8);
      }
    }
  }
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
