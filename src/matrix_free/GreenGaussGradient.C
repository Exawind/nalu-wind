// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradient.h"

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinSysInfo.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraMap.h"
#include "stk_mesh/base/GetNgpField.hpp"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <ostream>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
GreenGaussGradient<p>::GreenGaussGradient(
  stk::mesh::BulkData& bulk,
  Teuchos::ParameterList params,
  stk::mesh::Selector active,
  stk::mesh::Selector sides,
  stk::mesh::Selector replicas,
  Kokkos::View<gid_type*> rgids)
  : bulk_(bulk),
    active_(active),
    meta_(bulk.mesh_meta_data()),
    linsys_(
      stk::mesh::get_updated_ngp_mesh(bulk),
      active,
      linsys_info::get_gid_field(meta_),
      replicas,
      rgids),
    exporter_(
      Teuchos::rcpFromRef(linsys_.owned_and_shared),
      Teuchos::rcpFromRef(linsys_.owned)),
    conn_(
      stk_connectivity_map<p>(stk::mesh::get_updated_ngp_mesh(bulk), active)),
    offsets_(create_offset_map<p>(
      stk::mesh::get_updated_ngp_mesh(bulk),
      active,
      linsys_.stk_lid_to_tpetra_lid)),
    face_conn_(face_node_map<p>(stk::mesh::get_updated_ngp_mesh(bulk), sides)),
    bc_faces_(face_offsets<p>(
      stk::mesh::get_updated_ngp_mesh(bulk),
      sides,
      linsys_.stk_lid_to_tpetra_lid)),
    grad_(
      params, meta_, linsys_, exporter_, conn_, offsets_, face_conn_, bc_faces_)
{
}

template <int p>
void
GreenGaussGradient<p>::gradient(
  const stk::mesh::NgpField<double>& q, stk::mesh::NgpField<double>& dq)
{
  grad_.gradient(stk::mesh::get_updated_ngp_mesh(bulk_), active_, q, dq);
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
    elid_(linsys.stk_lid_to_tpetra_lid),
    q_(scalar_view<p>("q", offsets.extent_int(0))),
    dqdx_(vector_view<p>("dqdx", offsets.extent_int(0))),
    face_q_(face_scalar_view<p>("face_q", bc_faces.extent_int(0)))
{
  {
    auto coords = vector_view<p>("coords", conn_.extent_int(0));
    field_gather<p>(
      conn_, stk::mesh::get_updated_ngp_field<double>(*meta.coordinate_field()),
      coords);

    vols_ = geom::volume_metric<p>(coords);
    areas_ = geom::linear_areas<p>(coords);
  }

  {
    auto face_coords = face_vector_view<p>("coords", face_conn_.extent_int(0));
    field_gather<p>(
      face_conn_,
      stk::mesh::get_updated_ngp_field<double>(*meta.coordinate_field()),
      face_coords);

    exposed_areas_ = geom::exposed_areas<p>(face_coords);
  }
  update_.compute_preconditioner(vols_);
}

template <int p>
void
ComputeGradient<p>::gradient(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  const stk::mesh::NgpField<double>& q,
  stk::mesh::NgpField<double>& dqdx)
{
  field_gather<p>(conn_, q, q_);
  field_gather<p>(conn_, dqdx, dqdx_);
  field_gather<p>(face_conn_, q, face_q_);

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
    mesh, sel, elid_, delta.getLocalViewDevice(Tpetra::Access::ReadOnly), dqdx);
}

INSTANTIATE_POLYCLASS(ComputeGradient);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
