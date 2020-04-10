#include "matrix_free/StkEntityToRowMap.h"

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <stk_util/util/StkNgpVector.hpp>
#include <unordered_map>
#include <utility>

#include "Kokkos_Core.hpp"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/HashEntityAndEntityKey.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_topology/topology.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

constexpr int invalid_lid = -1;

entity_row_view_type
entity_to_row_lid_mapping(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Field<stk::mesh::EntityId>& gid_field,
  const stk::mesh::Selector& active,
  const std::unordered_map<stk::mesh::EntityId, int>& gid_to_lid)
{
  entity_row_view_type elid(
    "entityToLID", bulk.get_size_of_entity_index_space());
  const auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, invalid_lid);
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto ent : *ib) {
      const auto id = *stk::mesh::field_data(gid_field, ent);
      const auto iter = gid_to_lid.find(id);
      if (iter != gid_to_lid.end()) {
        elid_h(ent.local_offset()) = iter->second;
        if (id != bulk.identifier(ent)) {
          elid_h(bulk.get_entity(stk::topology::NODE_RANK, id).local_offset()) =
            elid_h(ent.local_offset());
        }
      }
    }
  }
  Kokkos::deep_copy(elid, elid_h);
  return elid;
}

entity_row_view_type
entity_to_row_lid_mapping(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& stk_gid_field,
  const stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
    tpetra_gid_field,
  const stk::mesh::Selector& active)
{
  return entity_to_row_lid_mapping(
    mesh.get_bulk_on_host(), stk_gid_field, active,
    global_to_local_id_map(mesh, stk_gid_field, tpetra_gid_field, active));
}

namespace {

int
count_valid_elids(const_entity_row_view_type elid)
{
  int count = 0;
  Kokkos::parallel_reduce(
    elid.extent_int(0),
    KOKKOS_LAMBDA(int k, int& valid_count) {
      valid_count += static_cast<int>(elid(k) != invalid_lid);
    },
    count);
  return count;
}

stk::NgpVector<int>
valid_elid_offsets(const_entity_row_view_type elid)
{
  // FIXME: parallel scan rather than doing thing on host
  const int num_valid = count_valid_elids(elid);
  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);

  stk::NgpVector<int> offsets(num_valid);
  int offset_index = 0;
  for (int k = 0; k < elid_h.extent_int(0); ++k) {
    if (elid_h(k) != invalid_lid) {
      offsets[offset_index++] = k;
    }
  }
  offsets.copy_host_to_device();
  return offsets;
}

} // namespace

mesh_index_row_view_type
row_lid_to_mesh_index_mapping(
  const stk::mesh::NgpMesh& mesh, const const_entity_row_view_type elid)
{
  const auto valid_elids = valid_elid_offsets(elid);
  auto lide = mesh_index_row_view_type("lide", valid_elids.size());
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
