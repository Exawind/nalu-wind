// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_TO_ENTITY_ROW_MAP_H
#define STK_TO_ENTITY_ROW_MAP_H

#include <Kokkos_View.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "Kokkos_Core.hpp"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Ngp.hpp"

#include "Tpetra_Map.hpp"

namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

entity_row_view_type entity_to_row_lid_mapping(
  const stk::mesh::BulkData&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  const stk::mesh::Selector&,
  const std::unordered_map<stk::mesh::EntityId, int>&);

entity_row_view_type entity_to_row_lid_mapping(
  const stk::mesh::NgpMesh&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  const stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&,
  const stk::mesh::Selector&);

mesh_index_row_view_type row_lid_to_mesh_index_mapping(
  const stk::mesh::NgpMesh&, const const_entity_row_view_type elid);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
