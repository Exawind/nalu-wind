// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOWMACH_GATHERED_FIELD_MANAGER_H
#define LOWMACH_GATHERED_FIELD_MANAGER_H

#include "matrix_free/LowMachFields.h"
#include "matrix_free/KokkosViewTypes.h"

#include "matrix_free/LowMachInfo.h"
#include "matrix_free/TransportCoefficients.h"

#include "stk_mesh/base/Selector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class LowMachGatheredFieldManager
{
public:
  using info = lowmach_info;
  LowMachGatheredFieldManager(
    stk::mesh::BulkData&, stk::mesh::Selector, stk::mesh::Selector = {});
  void gather_all();
  void update_fields();
  void swap_states();

  LowMachResidualFields<p> get_residual_fields() const { return fields; }
  LowMachLinearizedResidualFields<p> get_coefficient_fields() const
  {
    return coefficient_fields;
  }
  LowMachBCFields<p> get_bc_fields() const { return bc; }

  void update_mdot(double scaling);
  void update_pressure();
  void update_velocity();
  void update_grad_p();
  void update_transport_coefficients(GradTurbModel model);

private:
  stk::mesh::BulkData& bulk;
  const stk::mesh::MetaData& meta;
  const stk::mesh::Selector active;
  const stk::mesh::Selector dirichlet;
  const const_elem_mesh_index_view<p> conn;
  const const_face_mesh_index_view<p> exposed_faces;
  const const_node_mesh_index_view dirichlet_nodes;
  
  LowMachResidualFields<p> fields;
  LowMachLinearizedResidualFields<p> coefficient_fields;
  LowMachBCFields<p> bc;

  scalar_view<p> filter_scale;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
