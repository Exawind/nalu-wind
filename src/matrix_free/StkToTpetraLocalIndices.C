// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/KokkosFramework.h"

#include "Tpetra_Map.hpp"

#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpForEachEntity.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

Kokkos::View<typename Tpetra::Map<>::local_ordinal_type*>
make_stk_lid_to_tpetra_lid_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_in_mesh,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids,
  const Tpetra::Map<>::local_map_type& local_oas_map)
{
  Kokkos::View<typename Tpetra::Map<>::local_ordinal_type*> elid(
    Kokkos::ViewAllocateWithoutInitializing("entity_to_lid"),
    mesh.get_bulk_on_host().get_size_of_entity_index_space());
  Kokkos::deep_copy(elid, invalid_lid);
  exec_space().fence();

  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active_in_mesh,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const auto ent = mesh.get_entity(stk::topology::NODE_RANK, mi);
      elid(ent.local_offset()) = local_oas_map.getLocalElement(gids.get(mi, 0));
    });
  return elid;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
