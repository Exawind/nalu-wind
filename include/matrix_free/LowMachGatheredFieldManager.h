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

#include "stk_mesh/base/Selector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class LowMachGatheredFieldManager
{
public:
  LowMachGatheredFieldManager(stk::mesh::BulkData&, stk::mesh::Selector);
  void gather_all();
  void update_fields();
  void swap_states();

  LowMachResidualFields<p> get_residual_fields() const { return fields; }
  LowMachLinearizedResidualFields<p> get_coefficient_fields() const
  {
    return coefficient_fields;
  }

  void update_mdot(double scaling);
  void update_pressure();
  void update_velocity();
  void update_grad_p();
  void update_transport_coefficients();

private:
  stk::mesh::BulkData& bulk;
  const stk::mesh::MetaData& meta;
  const stk::mesh::Selector active;
  const const_elem_mesh_index_view<p> conn;
  LowMachResidualFields<p> fields;
  LowMachLinearizedResidualFields<p> coefficient_fields;

  scalar_view<p> scratch_volume_metric;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
