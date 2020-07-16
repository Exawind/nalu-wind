#include "matrix_free/GreenGaussGradientOperator.h"

#include "matrix_free/GreenGaussGradientInterior.h"
#include "matrix_free/GreenGaussBoundaryClosure.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"

#include "Tpetra_Operator.hpp"
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
    cached_shared_rhs_(exporter_in.getSourceMap(), num_vectors)
{
}

template <int p>
void
GradientResidualOperator<p>::compute(mv_type& owned_rhs)
{
  stk::mesh::ProfilingBlock pf("GradientResidualOperator<p>::apply");
  cached_shared_rhs_.putScalar(0.);
  gradient_residual<p>(
    elem_offsets_, residual_fields_.areas, residual_fields_.vols,
    residual_fields_.q, residual_fields_.dqdx,
    cached_shared_rhs_.getLocalViewDevice());

  if (face_bc_active_) {
    gradient_boundary_closure<p>(
      face_bc_offsets_, face_q_, exposed_areas_,
      cached_shared_rhs_.getLocalViewDevice());
  }

  cached_shared_rhs_.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.modify_device();
  owned_rhs.doExport(cached_shared_rhs_, exporter_, Tpetra::ADD);
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
  ThrowRequire(owned_sln.getNumVectors() == 3);
  stk::mesh::ProfilingBlock pf("GradientLinearizedResidualOperator<p>::apply");
  cached_sln_.doImport(owned_sln, exporter_, Tpetra::INSERT);
  cached_rhs_.putScalar(0.);

  filter_linearized_residual<p>(
    elem_offsets_, volumes_, cached_sln_.getLocalViewDevice(),
    cached_rhs_.getLocalViewDevice());

  cached_rhs_.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.modify_device();
  exec_space().fence();
  owned_rhs.doExport(cached_rhs_, exporter_, Tpetra::ADD);
}
INSTANTIATE_POLYCLASS(GradientLinearizedResidualOperator);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra