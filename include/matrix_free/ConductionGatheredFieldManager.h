#ifndef CONDUCTION_GATHERED_FIELD_MANAGER_H
#define CONDUCTION_GATHERED_FIELD_MANAGER_H

#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosFramework.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ConductionGatheredFieldManager
{
public:
  ConductionGatheredFieldManager(
    stk::mesh::BulkData&,
    stk::mesh::Selector,
    stk::mesh::Selector = {},
    stk::mesh::Selector = {});

  void gather_all();
  void update_solution_fields();
  void swap_states();

  InteriorResidualFields<p> get_residual_fields() { return fields; }
  BCDirichletFields get_bc_fields() { return bc_fields; }
  LinearizedResidualFields<p> get_coefficient_fields()
  {
    return coefficient_fields;
  }
  BCFluxFields<p> get_flux_fields() { return flux_fields; }

private:
  stk::mesh::BulkData& bulk;
  const stk::mesh::MetaData& meta;

  const stk::mesh::Selector active;
  const const_elem_mesh_index_view<p> conn;
  InteriorResidualFields<p> fields;
  LinearizedResidualFields<p> coefficient_fields;

  const stk::mesh::Selector dirichlet;
  const const_node_mesh_index_view dirichlet_nodes;
  BCDirichletFields bc_fields;

  const stk::mesh::Selector flux;
  const const_face_mesh_index_view<p> flux_faces;
  BCFluxFields<p> flux_fields;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
