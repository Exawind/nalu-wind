// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradientOperator.h"

#include "matrix_free/GreenGaussGradientInterior.h"
#include "matrix_free/GreenGaussBoundaryClosure.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_util/util/ReportHandler.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
GradientResidualOperator<p>::GradientResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
GradientResidualOperator<p>::compute(mv_type& owned_rhs)
{
  stk::mesh::ProfilingBlock pf("GradientResidualOperator<p>::apply");
  if (exporter_.getTargetMap()->isDistributed()) {

    cached_rhs_.putScalar(0.);
    gradient_residual<p>(
      elem_offsets_, residual_fields_.areas, residual_fields_.vols,
      residual_fields_.q, residual_fields_.dqdx,
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    if (face_bc_active_) {
      gradient_boundary_closure<p>(
        face_bc_offsets_, face_q_, exposed_areas_,
        cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }
    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    owned_rhs.putScalar(0.);
    gradient_residual<p>(
      elem_offsets_, residual_fields_.areas, residual_fields_.vols,
      residual_fields_.q, residual_fields_.dqdx,
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite), false);

    if (face_bc_active_) {
      gradient_boundary_closure<p>(
        face_bc_offsets_, face_q_, exposed_areas_,
        owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
    }
  }
}

INSTANTIATE_POLYCLASS(GradientResidualOperator);

template <int p>
GradientLinearizedResidualOperator<p>::GradientLinearizedResidualOperator(
  const_elem_offset_view<p> elem_offsets_in, const export_type& exporter_in)
  : elem_offsets_(elem_offsets_in),
    exporter_(exporter_in),
    cached_sln_(exporter_in.getSourceMap(), num_vectors),
    cached_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
GradientLinearizedResidualOperator<p>::apply(
  const mv_type& owned_sln,
  mv_type& owned_rhs,
  Teuchos::ETransp,
  double,
  double) const
{
  stk::mesh::ProfilingBlock pf("GradientLinearizedResidualOperator<p>::apply");

  STK_ThrowRequire(owned_sln.getNumVectors() == 3);

  if (exporter_.getTargetMap()->isDistributed()) {
    {
      stk::mesh::ProfilingBlock pfinner("import into owned-shared from owned");
      cached_sln_.doImport(owned_sln, exporter_, Tpetra::INSERT);
    }
    cached_rhs_.putScalar(0.);

    filter_linearized_residual<p>(
      elem_offsets_, volumes_,
      cached_sln_.getLocalViewDevice(Tpetra::Access::ReadWrite),
      cached_rhs_.getLocalViewDevice(Tpetra::Access::ReadWrite));

    owned_rhs.putScalar(0.);
    {
      stk::mesh::ProfilingBlock pfinner("export from owned-shared to owned");
      owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
    }
  } else {
    owned_rhs.putScalar(0.);
    filter_linearized_residual<p>(
      elem_offsets_, volumes_,
      owned_sln.getLocalViewDevice(Tpetra::Access::ReadOnly),
      owned_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  }
}
INSTANTIATE_POLYCLASS(GradientLinearizedResidualOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra