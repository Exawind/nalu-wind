// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkToTpetraLocalIndices.h"
#include "Tpetra_Map.hpp"

#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*>
make_stk_lid_to_tpetra_lid_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_in_mesh,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids,
  const Tpetra::Map<>::local_map_type& local_oas_map)
{
  Kokkos::View<typename Tpetra::Map<>::local_ordinal_type*> elid(
    "entity_to_lid", mesh.get_bulk_on_host().get_size_of_entity_index_space());
  Kokkos::deep_copy(elid, -1);

  const auto buckets =
    mesh.get_bucket_ids(stk::topology::NODE_RANK, active_in_mesh);
  Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(buckets.size(), Kokkos::AUTO),
    KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<>::member_type& team) {
      const auto league_index = team.league_rank();
      const auto bucket_id = buckets.device_get(league_index);
      const auto& b = mesh.get_bucket(stk::topology::NODE_RANK, bucket_id);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, b.size()), [&](int k) {
        const auto ent = b[k];
        auto fmi = mesh.fast_mesh_index(ent);
        elid(ent.local_offset()) =
          local_oas_map.getLocalElement(gids.get(fmi, 0));
      });
    });
  return elid;
}

namespace {

int
count_valid_elids(Kokkos::View<const int*> elid)
{
  int count = 0;
  Kokkos::parallel_reduce(
    elid.extent_int(0),
    KOKKOS_LAMBDA(int k, int& valid_count) {
      valid_count += static_cast<int>(elid(k) != -1);
    },
    count);
  return count;
}

stk::NgpVector<int>
valid_elid_offsets(
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid)
{
  // FIXME: don't do thing on host
  const int num_valid = count_valid_elids(elid);
  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);

  stk::NgpVector<int> offsets(num_valid);
  int offset_index = 0;
  for (int k = 0; k < elid_h.extent_int(0); ++k) {
    if (elid_h(k) != -1) {
      offsets[offset_index++] = k;
    }
  }
  offsets.copy_host_to_device();
  return offsets;
}

} // namespace

Kokkos::View<const stk::mesh::FastMeshIndex*>
make_tpetra_lid_to_stk_lid(
  const stk::mesh::NgpMesh& mesh,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid)
{
  const auto valid_elids = valid_elid_offsets(elid);
  Kokkos::View<stk::mesh::FastMeshIndex*> lide("lide", valid_elids.size());
  Kokkos::parallel_for(
    lide.extent_int(0), KOKKOS_LAMBDA(int k) {
      const stk::mesh::Entity ent =
        stk::mesh::Entity(valid_elids.device_get(k));
      const int lid = elid(ent.local_offset());
      lide(lid) = mesh.fast_mesh_index(ent);
    });
  return lide;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
