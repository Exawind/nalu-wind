// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionFields.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"

#include <stdexcept>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
InteriorResidualFields<p>
gather_required_conduction_fields_t<p>::invoke(
  const stk::mesh::MetaData& meta, const_elem_mesh_index_view<p> conn)
{
  InteriorResidualFields<p> fields;

  fields.qp1 = scalar_view<p>{"qp1", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNP1),
    fields.qp1);
  fields.qp0 = scalar_view<p>{"qp0", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateN),
    fields.qp0);
  fields.qm1 = scalar_view<p>{"qm1", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNM1),
    fields.qm1);

  vector_view<p> coords{"coords", conn.extent(0)};
  stk_simd_vector_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::coord_name), coords);

  scalar_view<p> alpha{"alpha", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::volume_weight_name), alpha);
  fields.volume_metric = geom::volume_metric<p>(alpha, coords);

  scalar_view<p> lambda{"lambda", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field(meta, conduction_info::diffusion_weight_name), lambda);
  fields.diffusion_metric = geom::diffusion_metric<p>(lambda, coords);

  return fields;
}
INSTANTIATE_POLYSTRUCT(gather_required_conduction_fields_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
