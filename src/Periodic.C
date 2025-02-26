/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#include "Periodic.h"

#include "stk_search_util/PeriodicBoundarySearch.hpp"

#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "stk_mesh/base/Selector.hpp"

namespace sierra::nalu::periodic {

namespace {
std::pair<bool, std::ostringstream>
check_for_periodic_matching_errors(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& sel,
  const stk::mesh::Field<double>& coord_field,
  const stk::mesh::PeriodicBoundarySearch<
    stk::mesh::GetCoordinates<stk::mesh::Field<double>>>& search)
{

  bool has_error = false;
  std::ostringstream error_msg;
  const auto& node_pairs = search.get_pairs();

  coord_field.sync_to_host();
  for (const auto& ib : bulk.get_buckets(stk::topology::NODE_RANK, sel)) {
    for (const auto& node : *ib) {
      const auto id = bulk.identifier(node);
      if (!std::any_of(
            node_pairs.begin(), node_pairs.end(), [&id](const auto& entities) {
              return (id == entities.first.id() || id == entities.second.id());
            })) {
        has_error = true;
      }

      if (has_error) {
        const auto* coords = stk::mesh::field_data(coord_field, node);
        error_msg << bulk.entity_key(node) << " (" << coords[0] << ", "
                  << coords[1] << ", " << coords[2] << ")" << std::endl;
      }
    }
  }
  return {has_error, std::move(error_msg)};
}

Kokkos::View<
  const Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>*,
  HostSpace>
copy_to_view(
  const std::vector<Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>>& pairs)
{
  auto ents = Kokkos::View<
    Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>*, HostSpace>(
    "host/device entities", pairs.size());
  for (int k = 0; k < ents.extent_int(0); ++k) {
    ents(k) = pairs[k];
  }
  return ents;
}

} // namespace

std::optional<periodic::TranslationMapping>
make_translation_mapping(
  const std::vector<PeriodicBCData>& translation_periodic_bcs,
  stk::mesh::BulkData& mesh,
  const stk::mesh::Selector& active_not_aura_selector,
  const std::string& coord_name)
{
  if (translation_periodic_bcs.empty()) {
    return std::nullopt;
  }

  const auto& meta = mesh.mesh_meta_data();
  STK_ThrowAssert(meta.get_field<double>(stk::topology::NODE_RANK, coord_name));

  const auto& coord_fld =
    *meta.get_field<double>(stk::topology::NODE_RANK, coord_name);
  stk::mesh::PeriodicBoundarySearch<
    stk::mesh::GetCoordinates<stk::mesh::Field<double>>>
    search(mesh, {mesh, coord_fld});

  stk::mesh::Selector full_periodic_sel;
  for (const auto& periodic_bc : translation_periodic_bcs) {
    const auto sel_a_bc = *periodic_bc.part_a & active_not_aura_selector;
    const auto sel_b_bc = *periodic_bc.part_b & active_not_aura_selector;
    search.add_linear_periodic_pair(sel_a_bc, sel_b_bc, periodic_bc.search_tol);
    STK_ThrowRequire(!periodic_bc.rotational);
    full_periodic_sel |= sel_a_bc;
    full_periodic_sel |= sel_b_bc;
  }
  search.find_periodic_nodes(mesh.parallel());
  stk::mesh::Selector sel_a =
    search.get_domain_selector() & active_not_aura_selector;
  stk::mesh::Selector sel_b =
    search.get_range_selector() & active_not_aura_selector;

  if (mesh.parallel_size() > 1) {
    mesh.modification_begin();
    search.create_ghosting(periodic::ghosting_name);
    mesh.modification_end();
  }

  const auto& [err, err_msg] = check_for_periodic_matching_errors(
    mesh, full_periodic_sel, coord_fld, search);
  std::ostringstream periodic_errors;
  if (err) {
    auto transforms = search.get_transforms();
    std::ostringstream msg;
    msg << "Periodic translation vectors:\n";
    for (const auto& t : transforms) {
      if (t.m_transform_type == decltype(search)::TRANSLATION) {
        msg << "  v = " << t.m_translation[0] << ", " << t.m_translation[1]
            << ", " << t.m_translation[2] << "\n";
      }
    }

    periodic_errors << "Periodic BCs could not match all nodes.\n"
                    << msg.str() << "\nProblem nodes are:\n"
                    << err_msg.str() << "\n";
  }
  if (err) {
    throw std::runtime_error(
      "Periodic BC errors encountered matching nodes\n" +
      periodic_errors.str());
    return std::nullopt;
  }

  using ent_pair_t = Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>;
  std::vector<ent_pair_t> non_ghost_a_pairs;
  std::vector<ent_pair_t> non_ghost_b_pairs;
  non_ghost_a_pairs.reserve(search.size());
  non_ghost_b_pairs.reserve(search.size());
  for (size_t k = 0; k < search.size(); ++k) {
    auto [node_a, node_b] = search.get_node_pair(k);

    const auto& a_bucket = mesh.bucket(node_a);
    if (a_bucket.owned() || a_bucket.shared()) {
      non_ghost_a_pairs.emplace_back(node_a, node_b);
    }

    const auto& b_bucket = mesh.bucket(node_b);
    if (b_bucket.owned() || b_bucket.shared()) {
      non_ghost_b_pairs.emplace_back(node_a, node_b);
    }
  }
  auto local_a_ents_h = copy_to_view(non_ghost_a_pairs);
  auto local_b_ents_h = copy_to_view(non_ghost_b_pairs);
  Kokkos::fence();

  auto local_a =
    Kokkos::create_mirror_view_and_copy(DeviceSpace{}, local_a_ents_h);
  auto local_b =
    Kokkos::create_mirror_view_and_copy(DeviceSpace{}, local_b_ents_h);
  return std::make_optional<periodic::TranslationMapping>(
    {local_a, local_b, sel_a, sel_b});
}

namespace {

constexpr bool force_atomic = !std::is_same_v<DeviceSpace, Kokkos::Serial>;

template <typename T>
struct copy_into_b_op
{
  KOKKOS_FUNCTION void operator()(const T& a, T& b) const
  {
    if constexpr (force_atomic) {
      Kokkos::atomic_store(&b, a);
    } else {
      b = a;
    }
  }
};

template <typename T>
struct sum_into_a_op
{
  KOKKOS_FUNCTION void operator()(T& a, const T& b) const
  {
    if constexpr (force_atomic) {
      Kokkos::atomic_add(&a, b);
    } else {
      a += b;
    }
  }
};

template <typename T>
struct max_into_a_op
{
  KOKKOS_FUNCTION void operator()(T& a, const T& b) const
  {
    if constexpr (force_atomic) {
      Kokkos::atomic_max(&a, b);
    } else {
      a = Kokkos::max(a, b);
    }
  }
};

template <typename T, typename Op>
void
local_periodic_op(
  const DeviceSpace& exec,
  const stk::mesh::NgpMesh& mesh,
  const periodic::TranslationMapping::map_t& pairs,
  stk::mesh::NgpField<T> field,
  Op&& op)
{
  field.sync_to_device();
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(exec, 0, pairs.extent_int(0)), KOKKOS_LAMBDA(int k) {
      const auto& pair = pairs(k);
      const auto& a_index = mesh.fast_mesh_index(pair.first);
      const auto& b_index = mesh.fast_mesh_index(pair.second);
      const int dim = int(field.get_num_components_per_entity(a_index));
      STK_NGP_ThrowAssert(
        dim == int(field.get_num_components_per_entity(b_index)));
      for (int d = 0; d < dim; ++d) {
        op(field(a_index, d), field(b_index, d));
      }
    });
  field.modify_on_device();
}

template <typename T, typename Op>
void
periodic_op(
  const DeviceSpace& exec,
  const stk::mesh::NgpMesh& mesh,
  const periodic::TranslationMapping::map_t& pairs,
  stk::mesh::NgpField<T> f,
  bool local,
  Op&& op)
{
  if (!local && mesh.get_bulk_on_host().parallel_size() > 1) {
    stk::mesh::communicate_field_data<T>(
      get_ghosting(mesh.get_bulk_on_host()), {&f});
  }
  local_periodic_op<T>(exec, mesh, pairs, f, op);
}

template <typename T>
void
sync(
  const DeviceSpace& exec,
  const periodic::TranslationMapping& data,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local)
{
  STK_ThrowRequire(field.type_is<T>());
  STK_ThrowRequire(field.entity_rank() == stk::topology::NODE_RANK);
  periodic_op<T>(
    exec, stk::mesh::get_updated_ngp_mesh(bulk), data.local_b_pairs,
    stk::mesh::get_updated_ngp_field<T>(field), local, copy_into_b_op<T>{});
}

template <typename T>
void
sum(
  const DeviceSpace& exec,
  const periodic::TranslationMapping& data,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local)
{
  STK_ThrowRequire(field.type_is<T>());
  STK_ThrowRequire(field.entity_rank() == stk::topology::NODE_RANK);
  periodic_op<T>(
    exec, stk::mesh::get_updated_ngp_mesh(bulk), data.local_a_pairs,
    stk::mesh::get_updated_ngp_field<T>(field), local, sum_into_a_op<T>{});
  exec.fence();
  sync<T>(exec, data, bulk, field, local);
}

template <typename T>
void
max(
  const DeviceSpace& exec,
  const periodic::TranslationMapping& data,
  const stk::mesh::BulkData& bulk,
  stk::mesh::FieldBase& field,
  bool local)
{
  STK_ThrowRequire(field.type_is<T>());
  STK_ThrowRequire(field.entity_rank() == stk::topology::NODE_RANK);
  periodic_op<T>(
    exec, stk::mesh::get_updated_ngp_mesh(bulk), data.local_a_pairs,
    stk::mesh::get_updated_ngp_field<T>(field), local, max_into_a_op<T>{});
  exec.fence();
  sync<T>(exec, data, bulk, field, local);
}

} // namespace

void
set_periodic_on_mesh(
  stk::mesh::BulkData& mesh,
  const std::optional<periodic::TranslationMapping>& periodic)
{
  const auto* periodic_ptr =
    periodic ? std::addressof(periodic.value()) : nullptr;
  mesh.mesh_meta_data()
    .declare_attribute_no_delete<periodic::TranslationMapping>(periodic_ptr);
}

void
sync(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& mesh,
  stk::mesh::FieldBase& field,
  bool local)
{
  const auto* periodic =
    mesh.mesh_meta_data().get_attribute<TranslationMapping>();

  if (!periodic) {
    return;
  }
  if (field.type_is<int>()) {
    sync<int>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<double>()) {
    sync<double>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<long long>()) {
    sync<long long>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<stk::mesh::EntityId>()) {
    sync<stk::mesh::EntityId>(exec, *periodic, mesh, field, local);
  } else {
    STK_ThrowErrorMsg("Invalid field type for field " + field.name());
  }
}

void
sum(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& mesh,
  stk::mesh::FieldBase& field,
  bool local)
{
  const auto* periodic =
    mesh.mesh_meta_data().get_attribute<TranslationMapping>();
  if (!periodic) {
    return;
  }
  if (field.type_is<int>()) {
    sum<int>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<double>()) {
    sum<double>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<long long>()) {
    sum<long long>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<stk::mesh::EntityId>()) {
    sum<stk::mesh::EntityId>(exec, *periodic, mesh, field, local);
  } else {
    STK_ThrowErrorMsg("Invalid field type for field " + field.name());
  }
}

void
max(
  const DeviceSpace& exec,
  const stk::mesh::BulkData& mesh,
  stk::mesh::FieldBase& field,
  bool local)
{
  const auto* periodic =
    mesh.mesh_meta_data().get_attribute<TranslationMapping>();
  if (!periodic) {
    return;
  }
  if (field.type_is<int>()) {
    max<int>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<double>()) {
    max<double>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<long long>()) {
    max<long long>(exec, *periodic, mesh, field, local);
  } else if (field.type_is<stk::mesh::EntityId>()) {
    max<stk::mesh::EntityId>(exec, *periodic, mesh, field, local);
  } else {
    STK_ThrowErrorMsg("Invalid field type for field " + field.name());
  }
}

stk::mesh::Ghosting&
get_ghosting(const stk::mesh::BulkData& bulk)
{
  const auto name = periodic::ghosting_name;
  const auto& ghostings = bulk.ghostings();
  auto it = std::find_if(
    ghostings.begin(), ghostings.end(),
    [&name](const auto* ghosting) { return ghosting->name() == name; });
  STK_ThrowRequireMsg(it != ghostings.end(), "No ghosting found");
  return **it;
}

stk::mesh::Selector
get_ghosting_selector(const stk::mesh::BulkData& bulk)
{
  if (bulk.parallel_size() == 1) {
    return {};
  }
  return bulk.ghosting_part(get_ghosting(bulk));
}

} // namespace sierra::nalu::periodic