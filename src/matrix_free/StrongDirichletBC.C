// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StrongDirichletBC.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosViewTypes.h"

#include <Teuchos_RCP.hpp>
#include <KokkosInterface.h>
#include <stk_simd/Simd.hpp>

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_util/util/ReportHandler.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

void
dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_scalar_view qp1,
  const_node_scalar_view qbc,
  int max_owned_row_lid,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("scalar_dirichlet_residual");
  Kokkos::parallel_for(
    "scalar_dirichlet_residual",
    DeviceRangePolicy(0, dirichlet_bc_offsets.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const auto residual = qbc(index) - qp1(index);
      const int valid_length = valid_offset(index, dirichlet_bc_offsets);
      for (int n = 0; n < valid_length; ++n) {
        const auto row_lid = dirichlet_bc_offsets(index, n);
        yout(row_lid, 0) =
          (row_lid < max_owned_row_lid) * stk::simd::get_data(residual, n);
      }
    });
}

void
dirichlet_residual(
  const_node_offset_view dirichlet_bc_offsets,
  const_node_vector_view qp1,
  const_node_vector_view qbc,
  int max_owned_row_lid,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("vector_dirichlet_residual");
  STK_ThrowRequireMsg(
    yout.extent_int(1) == 3, "length is " << yout.extent_int(1));

#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<2>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, int>;
#endif
  auto range = policy_type({0, 0}, {dirichlet_bc_offsets.extent_int(0), 3});
  Kokkos::parallel_for(
    range, KOKKOS_LAMBDA(int index, int d) {
      const auto residual = qbc(index, d) - qp1(index, d);
      const int valid_length = valid_offset(index, dirichlet_bc_offsets);
      for (int n = 0; n < valid_length; ++n) {
        const auto row_lid = dirichlet_bc_offsets(index, n);
        yout(row_lid, d) =
          (row_lid < max_owned_row_lid) * stk::simd::get_data(residual, n);
      }
    });
}

void
dirichlet_linearized(
  const_node_offset_view dirichlet_bc_offsets,
  int max_owned_row_lid,
  ra_tpetra_view_type xin,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("dirichlet_linearized");
  STK_ThrowRequire(yout.extent_int(0) == xin.extent_int(0));
  STK_ThrowRequire(yout.extent_int(1) == xin.extent_int(1));

#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<2>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, int>;
#endif
  auto range = policy_type(
    {0, 0}, {dirichlet_bc_offsets.extent_int(0), xin.extent_int(1)});
  Kokkos::parallel_for(
    range, KOKKOS_LAMBDA(int index, int d) {
      const int valid_length = valid_offset(index, dirichlet_bc_offsets);
      for (int n = 0; n < valid_length; ++n) {
        const auto row_lid = dirichlet_bc_offsets(index, n);
        yout(row_lid, d) = (row_lid < max_owned_row_lid) * xin(row_lid, d);
      }
    });
}

void
dirichlet_diagonal(
  const_node_offset_view offsets, int max_owned_lid, tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("dirichlet_diagonal");
#if defined(KOKKOS_ENABLE_HIP)
  using policy_type = Kokkos::MDRangePolicy<
    exec_space, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
    Kokkos::Rank<2>, int>;
#else
  using policy_type = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, int>;
#endif
  auto range = policy_type({0, 0}, {offsets.extent_int(0), yout.extent_int(1)});
  Kokkos::parallel_for(
    range, KOKKOS_LAMBDA(int index, int d) {
      const int valid_simd_len = valid_offset(index, offsets);
      for (int n = 0; n < valid_simd_len; ++n) {
        const auto row_lid = offsets(index, n);
        yout(row_lid, d) = int(row_lid < max_owned_lid);
      }
    });
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
