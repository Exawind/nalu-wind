#ifndef STKHELPERS_H
#define STKHELPERS_H

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_topology/topology.hpp>

#include "FieldTypeDef.h"

#include <array>

namespace sierra {
namespace nalu {

class Realm;

void populate_ghost_comm_procs(
  const stk::mesh::BulkData& bulk_data,
  stk::mesh::Ghosting& ghosting,
  std::vector<int>& ghostCommProcs);

stk::topology
get_elem_topo(const Realm& realm, const stk::mesh::Part& surfacePart);

void add_downward_relations(
  const stk::mesh::BulkData& bulk,
  std::vector<stk::mesh::EntityKey>& entityKeys);

void keep_elems_not_already_ghosted(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::EntityProcVec& alreadyGhosted,
  stk::mesh::EntityProcVec& elemsToGhost);

void fill_send_ghosts_to_remove_from_ghosting(
  const stk::mesh::EntityProcVec& curSendGhosts,
  const stk::mesh::EntityProcVec& intersection,
  stk::mesh::EntityProcVec& sendGhostsToRemove);

void communicate_to_fill_recv_ghosts_to_remove(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::EntityProcVec& sendGhostsToRemove,
  std::vector<stk::mesh::EntityKey>& recvGhostsToRemove);

void keep_only_elems(
  const stk::mesh::BulkData& bulk, stk::mesh::EntityProcVec& entityProcs);

void compute_precise_ghosting_lists(
  const stk::mesh::BulkData& bulk,
  stk::mesh::EntityProcVec& elemsToGhost,
  stk::mesh::EntityProcVec& curSendGhosts,
  std::vector<stk::mesh::EntityKey>& recvGhostsToRemove);

/** Return a field ordinal given the name of the field
 */
inline unsigned
get_field_ordinal(
  const stk::mesh::MetaData& meta,
  const std::string fieldName,
  const stk::mesh::EntityRank entity_rank = stk::topology::NODE_RANK)
{
  stk::mesh::FieldBase* field = meta.get_field(entity_rank, fieldName);
  ThrowRequireMsg(
    (field != nullptr), "Requested field does not exist: " + fieldName);
  return field->mesh_meta_data_ordinal();
}

/** Return a field ordinal for a particular state
 *
 */
inline unsigned
get_field_ordinal(
  const stk::mesh::MetaData& meta,
  const std::string fieldName,
  const stk::mesh::FieldState state,
  const stk::mesh::EntityRank entity_rank = stk::topology::NODE_RANK)
{
  const auto* field = meta.get_field(entity_rank, fieldName);
  ThrowRequireMsg(
    (field != nullptr), "Requested field does not exist: " + fieldName);
  ThrowRequireMsg(
    (field->is_state_valid(state)), "Requested invalid state: " + fieldName);

  const auto* fState = field->field_state(state);
  return fState->mesh_meta_data_ordinal();
}

template <typename T = double>
stk::mesh::NgpField<T>&
get_node_field(
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

// Can replace this function with field.max_extent(0) when
// using new-enough Trilinos.
//
unsigned max_extent(const stk::mesh::FieldBase& field, unsigned dimension);

} // namespace nalu
} // namespace sierra

#endif /* STKHELPERS_H */
