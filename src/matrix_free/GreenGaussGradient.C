// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradient.h"

#include "matrix_free/LinearVolume.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/LinSysInfo.h"

#include "stk_mesh/base/GetNgpField.hpp"

#include <Kokkos_ExecPolicy.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Ngp.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
GreenGaussGradient<p>::GreenGaussGradient(
  stk::mesh::BulkData& bulk,
  Teuchos::ParameterList params,
  stk::mesh::Selector active,
  stk::mesh::Selector sides,
  stk::mesh::Selector replicas)
  : meta_(bulk.mesh_meta_data()),
    linsys_(
      bulk.get_updated_ngp_mesh(),
      active,
      linsys_info::get_gid_field(meta_),
      replicas),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    conn_(stk_connectivity_map<p>(bulk.get_updated_ngp_mesh(), active)),
    offsets_(create_offset_map<p>(
      bulk.get_updated_ngp_mesh(), active, linsys_.stk_lid_to_tpetra_lid)),
    face_conn_(face_node_map<p>(bulk.get_updated_ngp_mesh(), sides)),
    bc_faces_(face_offsets<p>(
      bulk.get_updated_ngp_mesh(), sides, linsys_.stk_lid_to_tpetra_lid)),
    grad_(
      params, meta_, linsys_, exporter_, conn_, offsets_, face_conn_, bc_faces_)
{
}

template <int p>
void
GreenGaussGradient<p>::gradient(
  const stk::mesh::NgpField<double>& q, stk::mesh::NgpField<double>& dq)
{
  grad_.gradient(q, dq);
}

template <int p>
void
GreenGaussGradient<p>::banner(std::string name, std::ostream& stream) const
{
  const auto residual_norm = grad_.residual_norm();
  if (initial_residual_ < 0) {
    initial_residual_ = residual_norm;
  }
  const auto scaled_residual_norm =
    residual_norm /
    std::max(std::numeric_limits<double>::epsilon(), initial_residual_);

  const int nameOffset = name.length() + 8;
  stream << std::setw(nameOffset) << std::right << name
         << std::setw(32 - nameOffset) << std::right << grad_.num_iterations()
         << std::setw(18) << std::right << grad_.final_linear_norm()
         << std::setw(15) << std::right << residual_norm << std::setw(14)
         << std::right << scaled_residual_norm << std::endl;
}

INSTANTIATE_POLYCLASS(GreenGaussGradient);

template <int p>
ComputeGradient<p>::ComputeGradient(
  Teuchos::ParameterList params,
  const stk::mesh::MetaData& meta,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const_elem_mesh_index_view<p> conn,
  const_elem_offset_view<p> offsets,
  const_face_mesh_index_view<p> face_conn,
  const_face_offset_view<p> bc_faces)
  : update_(params, linsys, exporter, offsets, bc_faces),
    conn_(conn),
    face_conn_(face_conn),
    lide_(linsys.tpetra_lid_to_stk_lid),
    q_(scalar_view<p>("q", offsets.extent_int(0))),
    dqdx_(vector_view<p>("dqdx", offsets.extent_int(0))),
    face_q_(face_scalar_view<p>("face_q", bc_faces.extent_int(0)))
{
  {
    auto coords = vector_view<p>("coords", conn_.extent_int(0));
    stk_simd_vector_field_gather<p>(
      conn_, stk::mesh::get_updated_ngp_field<double>(*meta.coordinate_field()),
      coords);

    vols_ = geom::volume_metric<p>(coords);
    areas_ = geom::linear_areas<p>(coords);
  }

  {
    auto face_coords = face_vector_view<p>("coords", face_conn_.extent_int(0));
    stk_simd_face_vector_field_gather<p>(
      face_conn_,
      stk::mesh::get_updated_ngp_field<double>(*meta.coordinate_field()),
      face_coords);

    exposed_areas_ = geom::exposed_areas<p>(face_coords);
  }
  update_.compute_preconditioner(vols_);
}

namespace {
void
add_tpetra_solution_vector_to_stk_field(
  const_mesh_index_row_view_type lide,
  const typename Tpetra::MultiVector<>::dual_view_type::t_dev delta_view,
  stk::mesh::NgpField<double>& field)
{
  Kokkos::parallel_for(
    delta_view.extent_int(0), KOKKOS_LAMBDA(int k) {
      const auto fmi = lide(k);
      for (int d = 0; d < 3; ++d) {
        field.get(fmi, d) += delta_view(k, d);
      }
    });
  field.modify_on_device();
}
} // namespace

template <int p>
void
ComputeGradient<p>::gradient(
  const stk::mesh::NgpField<double>& q, stk::mesh::NgpField<double>& dqdx)
{
  stk_simd_scalar_field_gather<p>(conn_, q, q_);
  stk_simd_vector_field_gather<p>(conn_, dqdx, dqdx_);
  stk_simd_face_scalar_field_gather<p>(face_conn_, q, face_q_);

  GradientResidualFields<p> fields;
  fields.q = q_;
  fields.dqdx = dqdx_;
  fields.vols = vols_;
  fields.areas = areas_;

  BCGradientFields<p> bc_fields;
  bc_fields.face_q = face_q_;
  bc_fields.exposed_areas = exposed_areas_;

  update_.compute_residual(fields, bc_fields);
  const auto& delta = update_.compute_delta(vols_);
  add_tpetra_solution_vector_to_stk_field(
    lide_, delta.getLocalViewDevice(), dqdx);
}

INSTANTIATE_POLYCLASS(ComputeGradient);

template <int p>
GradientSolutionUpdate<p>::GradientSolutionUpdate(
  Teuchos::ParameterList params,
  const StkToTpetraMaps& linsys,
  const Tpetra::Export<>& exporter,
  const_elem_offset_view<p> offsets,
  const_face_offset_view<p> bc_faces)
  : linsys_(linsys),
    exporter_(exporter),
    offsets_(offsets),
    bc_faces_(bc_faces),
    resid_op_(offsets, exporter),
    lin_op_(offsets, exporter),
    prec_op_(offsets, exporter, 1),
    linear_solver_(lin_op_, 3, params),
    owned_and_shared_mv_(exporter.getSourceMap(), 3)
{
}

template <int p>
void
GradientSolutionUpdate<p>::compute_preconditioner(const_scalar_view<p> vols)
{
  stk::mesh::ProfilingBlock pf(
    "GradientSolutionUpdate<p>::::compute_preconditioner");
  prec_op_.compute_diagonal(vols);
  prec_op_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
  linear_solver_.set_preconditioner(prec_op_);
}

template <int p>
const Tpetra::MultiVector<double>&
GradientSolutionUpdate<p>::compute_residual(
  GradientResidualFields<p> fields, BCGradientFields<p> bc)
{
  resid_op_.set_fields(fields);
  resid_op_.set_bc_fields(bc_faces_, bc.exposed_areas, bc.face_q);
  resid_op_.compute(linear_solver_.rhs());
  return linear_solver_.rhs();
}

template <int p>
const Tpetra::MultiVector<double>&
GradientSolutionUpdate<p>::compute_delta(const_scalar_view<p> vols)
{
  stk::mesh::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_delta");

  lin_op_.set_volumes(vols);
  linear_solver_.solve();
  owned_and_shared_mv_.doImport(
    linear_solver_.lhs(), exporter_, Tpetra::INSERT);
  return owned_and_shared_mv_;
}

template <int p>
double
GradientSolutionUpdate<p>::residual_norm() const
{
  return linear_solver_.nonlinear_residual();
}
template <int p>
int
GradientSolutionUpdate<p>::num_iterations() const
{
  return linear_solver_.num_iterations();
}
template <int p>
double
GradientSolutionUpdate<p>::final_linear_norm() const
{
  return linear_solver_.final_linear_norm();;
}

INSTANTIATE_POLYCLASS(GradientSolutionUpdate);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
