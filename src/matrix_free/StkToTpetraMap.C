// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkToTpetraMap.h"

#include "matrix_free/StkToTpetraComm.h"
#include "matrix_free/StkToTpetraLocalIndices.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"
#include "Kokkos_Sort.hpp"

#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_topology/topology.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

void
add_tpetra_solution_vector_to_stk_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid,
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread
    delta_view,
  stk::mesh::NgpField<double>& field)
{
  stk::mesh::ProfilingBlock pf("add_tpetra_solution_vector_to_stk_field");

  const int dim = delta_view.extent_int(1);
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const auto ent = mesh.get_entity(stk::topology::NODE_RANK, mi);
      const auto tpetra_lid = elid(ent.local_offset());
      for (int d = 0; d < dim; ++d) {
        field(mi, d) += delta_view(tpetra_lid, d);
      }
    });
  field.modify_on_device();
}

StkToTpetraMaps::StkToTpetraMaps(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  stk::mesh::NgpField<gid_type> tgid,
  stk::mesh::Selector replicas,
  Kokkos::View<gid_type*> rgids)
  : owned(make_owned_row_map(mesh, active - replicas)),
    owned_and_shared(make_owned_shared_constrained_row_map(
      mesh, active - replicas, rgids, tgid)),
    stk_lid_to_tpetra_lid(make_stk_lid_to_tpetra_lid_map(
      mesh, active, tgid, owned_and_shared.getLocalMap()))
{
}

StkToTpetraMaps::StkToTpetraMaps(
  const Tpetra::Map<>& owned_in,
  const Tpetra::Map<>& owned_and_shared_in,
  Kokkos::View<const lid_type*> stk_lid_to_tpetra_lid_in)
  : owned(owned_in),
    owned_and_shared(owned_and_shared_in),
    stk_lid_to_tpetra_lid(stk_lid_to_tpetra_lid_in)
{
}

namespace {
const auto invalid_size_t =
  Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
using global_ordinal_type = typename Tpetra::Map<>::global_ordinal_type;

stk::NgpVector<int>
bucket_offsets(const stk::mesh::NgpMesh& mesh, stk::NgpVector<unsigned> buckets)
{
  stk::NgpVector<int> lengths(buckets.size());
  for (unsigned id = 0u; id < buckets.size(); ++id) {
    lengths[id] = mesh.get_bucket(stk::topology::NODE_RANK, buckets[id]).size();
  }
  stk::NgpVector<int> offset(buckets.size());
  int prev_sum = 0;
  for (unsigned k = 0u; k < buckets.size(); ++k) {
    offset[k] = prev_sum;
    prev_sum += lengths[k];
  }
  offset.copy_host_to_device();
  return offset;
}

template <typename Func>
void
enumerated_for_each_entity(
  const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& active, Func f)
{
  auto buckets = mesh.get_bucket_ids(stk::topology::NODE_RANK, active);
  auto offsets = bucket_offsets(mesh, buckets);
  Kokkos::parallel_for(
    DeviceTeamPolicy(buckets.size(), Kokkos::AUTO),
    KOKKOS_LAMBDA(const typename DeviceTeamPolicy::member_type& team) {
      const auto league_index = team.league_rank();
      const auto bucket_id = buckets.device_get(league_index);
      const auto& b = mesh.get_bucket(stk::topology::NODE_RANK, bucket_id);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, b.size()), [&](int k) {
        f(k + offsets.device_get(league_index), mesh.fast_mesh_index(b[k]));
      });
    });
}

size_t
count_local_nodes(
  const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& sel)
{
  const auto bucket_ids = mesh.get_bucket_ids(stk::topology::NODE_RANK, sel);
  size_t length = 0;
  for (unsigned k = 0u; k < bucket_ids.size(); ++k) {
    length += mesh.get_bucket(stk::topology::NODE_RANK, bucket_ids[k]).size();
  }
  return length;
}

} // namespace

void
populate_global_id_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids)
{
  const auto& bulk = mesh.get_bulk_on_host();
  const auto& meta = bulk.mesh_meta_data();
  const auto owned_selector = meta.locally_owned_part() & active_linsys;
  auto first_index =
    make_owned_row_map(mesh, active_linsys).getMinGlobalIndex();
  enumerated_for_each_entity(
    mesh, owned_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      gids.get(mi, 0) = first_index + index;
    });
  gids.modify_on_device();
  stk::mesh::communicate_field_data<global_ordinal_type>(
    mesh.get_bulk_on_host(), {&gids});
}

void
populate_global_id_field(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids)
{
  const auto& meta = bulk.mesh_meta_data();
  const auto owned_selector = meta.locally_owned_part() & active_linsys;
  auto first_index =
    make_owned_row_map(stk::mesh::get_updated_ngp_mesh(bulk), active_linsys)
      .getMinGlobalIndex();
  enumerated_for_each_entity(
    stk::mesh::get_updated_ngp_mesh(bulk), owned_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      gids.get(mi, 0) = first_index + index;
    });
  gids.modify_on_device();
  stk::mesh::communicate_field_data<global_ordinal_type>(bulk, {&gids});
}

Tpetra::Map<>
make_owned_row_map(
  const stk::mesh::NgpMesh& mesh, const stk::mesh::Selector& active_linsys)
{
  const auto& bulk = mesh.get_bulk_on_host();
  const auto& meta = bulk.mesh_meta_data();
  const auto owned_selector = meta.locally_owned_part() & active_linsys;
  const auto local_num_dofs = count_local_nodes(mesh, owned_selector);
  return Tpetra::Map<>(
    invalid_size_t, local_num_dofs, global_ordinal_index_base,
    teuchos_communicator(bulk.parallel()));
}

Tpetra::Map<>
make_owned_and_shared_row_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids)
{
  const auto& bulk = mesh.get_bulk_on_host();
  const auto& meta = bulk.mesh_meta_data();
  const auto owned_selector = meta.locally_owned_part() & active_linsys;
  const auto num_owned = count_local_nodes(mesh, owned_selector);
  const auto shared_selector =
    (meta.globally_shared_part() & active_linsys) - meta.locally_owned_part();
  const auto num_shared = count_local_nodes(mesh, shared_selector);

  STK_ThrowRequire(
    count_local_nodes(mesh, active_linsys) == num_owned + num_shared);
  Kokkos::View<global_ordinal_type*> row_ids(
    "oas_row_ids", count_local_nodes(mesh, active_linsys));
  enumerated_for_each_entity(
    mesh, owned_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      row_ids(index) = gids.get(mi, 0);
    });
  Kokkos::sort(row_ids, 0, num_owned);
  enumerated_for_each_entity(
    mesh, shared_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      row_ids(num_owned + index) = gids.get(mi, 0);
    });
  Kokkos::sort(row_ids, num_owned, row_ids.extent_int(0));
  return Tpetra::Map<>(
    invalid_size_t, row_ids, global_ordinal_index_base,
    teuchos_communicator(bulk.parallel()));
}

Tpetra::Map<>
make_owned_shared_constrained_row_map(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active_linsys,
  Kokkos::View<typename Tpetra::Map<>::global_ordinal_type*> rgids,
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gids)
{
  const auto& bulk = mesh.get_bulk_on_host();
  const auto& meta = bulk.mesh_meta_data();
  const auto owned_selector = meta.locally_owned_part() & active_linsys;
  const auto num_owned = count_local_nodes(mesh, owned_selector);
  const auto shared_selector =
    (meta.globally_shared_part() & active_linsys) - meta.locally_owned_part();
  const auto num_shared = count_local_nodes(mesh, shared_selector);

  const auto num_periodic_shared = rgids.extent_int(0);

  Kokkos::View<global_ordinal_type*> row_ids(
    "oas_row_ids", num_owned + num_shared + num_periodic_shared);
  enumerated_for_each_entity(
    mesh, owned_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      row_ids(index) = gids.get(mi, 0);
    });
  Kokkos::sort(row_ids, 0, num_owned);

  enumerated_for_each_entity(
    mesh, shared_selector,
    KOKKOS_LAMBDA(int index, stk::mesh::FastMeshIndex mi) {
      row_ids(num_owned + index) = gids.get(mi, 0);
    });
  Kokkos::parallel_for(
    DeviceRangePolicy(0, rgids.extent_int(0)),
    KOKKOS_LAMBDA(int k) { row_ids(num_owned + num_shared + k) = rgids(k); });
  Kokkos::sort(row_ids, num_owned, row_ids.extent_int(0));

  return Tpetra::Map<>(
    invalid_size_t, row_ids, global_ordinal_index_base,
    teuchos_communicator(bulk.parallel()));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
