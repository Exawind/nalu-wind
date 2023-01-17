// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COMPUTE_GRADIENT_H
#define COMPUTE_GRADIENT_H

#include "matrix_free/EquationUpdate.h"
#include "matrix_free/GradientSolutionUpdate.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinSysInfo.h"
#include "matrix_free/StkToTpetraMap.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"

#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <iosfwd>

namespace Teuchos {
class ParameterList;
}
namespace stk {
namespace mesh {
class BulkData;
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

// gradient with shared pre-processing
template <int p>
class ComputeGradient
{
public:
  ComputeGradient(
    Teuchos::ParameterList solver_params,
    const stk::mesh::MetaData& meta,
    const StkToTpetraMaps& linsys,
    const Tpetra::Export<>& exporter,
    const_elem_mesh_index_view<p> conn,
    const_elem_offset_view<p> offsets,
    const_face_mesh_index_view<p> face_conn,
    const_face_offset_view<p> bc_faces);

  void gradient(
    const stk::mesh::NgpMesh& mesh,
    const stk::mesh::Selector& sel,
    const stk::mesh::NgpField<double>& q,
    stk::mesh::NgpField<double>& dqdx);

  double residual_norm() const { return update_.residual_norm(); }
  double final_linear_norm() const { return update_.final_linear_norm(); }
  int num_iterations() const { return update_.num_iterations(); }

private:
  mutable GradientSolutionUpdate<p> update_;
  const const_elem_mesh_index_view<p> conn_;
  const const_face_mesh_index_view<p> face_conn_;
  const Kokkos::View<
    const typename Tpetra::Map<>::local_ordinal_type*,
    Kokkos::MemoryRandomAccess>
    elid_;

  scs_vector_view<p> areas_;
  face_vector_view<p> exposed_areas_;
  scalar_view<p> vols_;

  scalar_view<p> q_;
  vector_view<p> dqdx_;
  face_scalar_view<p> face_q_;
};

template <int p>
class GreenGaussGradient : public GradientUpdate
{
public:
  GreenGaussGradient(
    stk::mesh::BulkData&,
    Teuchos::ParameterList,
    stk::mesh::Selector active,
    stk::mesh::Selector sides,
    stk::mesh::Selector replicas = {},
    Kokkos::View<gid_type*> rgids = {});

  void gradient(
    const stk::mesh::NgpField<double>& q,
    stk::mesh::NgpField<double>& dqdx) final;

  void reset_initial_residual() final { initial_residual_ = -1; }
  void banner(std::string, std::ostream&) const final;

private:
  const stk::mesh::BulkData& bulk_;
  const stk::mesh::Selector active_;

  const stk::mesh::MetaData& meta_;
  const StkToTpetraMaps linsys_;
  const Tpetra::Export<> exporter_;
  const const_elem_mesh_index_view<p> conn_;
  const const_elem_offset_view<p> offsets_;
  const const_face_mesh_index_view<p> face_conn_;
  const const_face_offset_view<p> bc_faces_;
  mutable ComputeGradient<p> grad_;

  mutable double initial_residual_{-1};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
