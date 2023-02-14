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

#include "Tpetra_Map.hpp"

#include "Kokkos_Core.hpp"

#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

static constexpr int invalid_lid = -1;

Kokkos::View<typename Tpetra::Map<>::local_ordinal_type*>
make_stk_lid_to_tpetra_lid_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_in_mesh,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids,
  const Tpetra::Map<>::local_map_type& local_oas_map);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
