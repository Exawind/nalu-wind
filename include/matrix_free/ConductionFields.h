#ifndef CONDUCTION_FIELDS_H
#define CONDUCTION_FIELDS_H

#include "matrix_free/ConductionInfo.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

#include "Kokkos_Array.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

namespace stk {
namespace mesh {
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

template <typename T = double>
stk::mesh::NgpField<T>&
get_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

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
