// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_TO_TPETRA_LOCAL_INDICES_H
#define STK_TO_TPETRA_LOCAL_INDICES_H

#include "Tpetra_Map_decl.hpp"

#include "Kokkos_View.hpp"

#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*>
make_stk_lid_to_tpetra_lid_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_in_mesh,
  stk::mesh::NgpConstField<typename Tpetra::Map<>::global_ordinal_type> gids,
  const Tpetra::Map<>::local_map_type& local_oas_map);

Kokkos::View<const stk::mesh::FastMeshIndex*> make_tpetra_lid_to_stk_lid(
  const stk::mesh::NgpMesh& mesh,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
