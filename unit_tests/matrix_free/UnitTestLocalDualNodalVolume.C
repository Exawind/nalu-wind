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

#include "matrix_free/LocalDualNodalVolume.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/StkLowMachFixture.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/TensorOperations.h"

#include "stk_mesh/base/GetNgpField.hpp"

#include "StkSimdComparisons.h"
#include "gtest/gtest.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {
double
skew_mesh(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  stk::mesh::NgpField<double>& coords)
{
  ArrayND<double[3][3]> Q = {{{2, 1, 1.3333}, {0, 2, -1}, {1, 0, 2}}};
  double detq = determinant(Q);

  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, sel,
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
  return detq;
}
} // namespace

class LocalDualNodalVolumeFixture : public LowMachFixture
{
public:
  static constexpr int nx = 2;
  static constexpr double scaling = 2;

  LocalDualNodalVolumeFixture() : LowMachFixture(nx, scaling) {}
};

TEST_F(LocalDualNodalVolumeFixture, correct_volume_for_affine_block)
{
  if (bulk.parallel_size() > 1) {
    return;
  }

  auto coords = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  const double detq = skew_mesh(mesh(), active(), coords);

  auto dnv = stk::mesh::get_updated_ngp_field<double>(filter_scale_field);
  local_dual_nodal_volume(order, mesh(), active(), coords, dnv);
  dnv.sync_to_host();

  auto buckets = bulk.get_buckets(
    stk::topology::NODE_RANK, active() & meta.locally_owned_part());
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      EXPECT_NEAR(
        *stk::mesh::field_data(filter_scale_field, node),
        bulk.num_elements(node) * detq / 8, 1.0e-8);
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
