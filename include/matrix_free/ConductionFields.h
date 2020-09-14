// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_FIELDS_H
#define CONDUCTION_FIELDS_H

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/PolynomialOrders.h"
#include "stk_mesh/base/GetNgpField.hpp"

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/FieldState.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/util/ReportHandler.hpp>

#include <iosfwd>

namespace sierra {
namespace nalu {
namespace matrix_free {

struct BCDirichletFields
{
  node_scalar_view qp1;
  node_scalar_view qbc;
};

template <int p>
struct BCFluxFields
{
  face_scalar_view<p> flux;
  face_vector_view<p> exposed_areas;
};

template <int p>
struct BCFields
{
  BCDirichletFields dirichlet_fields;
  BCFluxFields<p> flux_fields;
};

template <int p>
struct InteriorResidualFields
{
  scalar_view<p> qm1;
  scalar_view<p> qp0;
  scalar_view<p> qp1;
  scalar_view<p> volume_metric;
  scs_vector_view<p> diffusion_metric;
};

template <int p>
struct LinearizedResidualFields
{
  scalar_view<p> volume_metric;
  scs_vector_view<p> diffusion_metric;
};

namespace impl {

template <int p>
struct gather_required_conduction_fields_t
{
  static InteriorResidualFields<p>
  invoke(const stk::mesh::MetaData&, const_elem_mesh_index_view<p>);
};

} // namespace impl
P_INVOKEABLE(gather_required_conduction_fields)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
