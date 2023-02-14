// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_SIMD_MESH_TRAVERSER_H
#define STK_SIMD_MESH_TRAVERSER_H

#include <KokkosInterface.h>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace stk {
namespace mesh {
struct Entity;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

KOKKOS_INLINE_FUNCTION int
bucket_index(int bktIndex, int simdElemIndex)
{
  return bktIndex * simd_len + simdElemIndex;
}

KOKKOS_INLINE_FUNCTION int
get_num_simd_groups(int length)
{
  return (length % simd_len > 0) ? length / simd_len + 1 : length / simd_len;
}

KOKKOS_INLINE_FUNCTION int
get_length_of_next_simd_group(int index, int length)
{
  int nextLength = simd_len;
  if (length - index * simd_len < simd_len) {
    nextLength = length - index * simd_len;
  }

  if (nextLength < 0 || nextLength > simd_len) {
    nextLength = 0;
  }
  return nextLength;
}

inline int
num_simd_elements(
  const stk::mesh::NgpMesh& mesh,
  stk::topology::rank_t rank,
  const stk::mesh::Selector& selector)
{
  const auto buckets = mesh.get_bucket_ids(rank, selector);
  int mesh_index = 0;
  for (unsigned id = 0u; id < buckets.size(); ++id) {
    mesh_index +=
      get_num_simd_groups(mesh.get_bucket(rank, buckets[id]).size());
  }
  return mesh_index;
}

inline stk::NgpVector<int>
simd_bucket_offsets(
  const stk::mesh::NgpMesh& mesh,
  stk::topology::rank_t rank,
  stk::NgpVector<unsigned> buckets)
{
  stk::NgpVector<int> simd_lengths(buckets.size());
  for (unsigned id = 0u; id < buckets.size(); ++id) {
    simd_lengths[id] =
      get_num_simd_groups(mesh.get_bucket(rank, buckets[id]).size());
  }
  stk::NgpVector<int> simd_offset(buckets.size());
  int prev_sum = 0;
  for (unsigned k = 0u; k < buckets.size(); ++k) {
    simd_offset[k] = prev_sum;
    prev_sum += simd_lengths[k];
  }
  simd_offset.copy_host_to_device();
  return simd_offset;
}

} // namespace impl

inline int
num_simd_elements(
  const stk::mesh::NgpMesh& mesh,
  stk::topology::rank_t rank,
  const stk::mesh::Selector& selector)
{
  const auto buckets = mesh.get_bucket_ids(rank, selector);
  int mesh_index = 0;
  for (unsigned id = 0u; id < buckets.size(); ++id) {
    mesh_index +=
      impl::get_num_simd_groups(mesh.get_bucket(rank, buckets[id]).size());
  }
  return mesh_index;
}

template <typename ValidFunc, typename RemainderFunc>
void
simd_traverse(
  const stk::mesh::NgpMesh& mesh,
  stk::topology::rank_t rank,
  const stk::mesh::Selector& active,
  ValidFunc func,
  RemainderFunc rem)
{

  auto buckets = mesh.get_bucket_ids(rank, active);
  const auto bucket_offsets = impl::simd_bucket_offsets(mesh, rank, buckets);
  Kokkos::parallel_for(
    DeviceTeamPolicy(buckets.size(), Kokkos::AUTO),
    KOKKOS_LAMBDA(const typename DeviceTeamPolicy::member_type& team) {
      const auto bucket_id = buckets.device_get(team.league_rank());
      const auto& b = mesh.get_bucket(rank, bucket_id);
      const auto bucket_len = b.size();
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, impl::get_num_simd_groups(bucket_len)),
        [&](int e) {
          const int num_simd_elems =
            impl::get_length_of_next_simd_group(e, bucket_len);
          const int simd_elem_index =
            bucket_offsets.device_get(team.league_rank()) + e;
          for (int ne = 0; ne < num_simd_elems; ++ne) {
            func(simd_elem_index, ne, b[impl::bucket_index(e, ne)]);
          }
          for (int ne = num_simd_elems; ne < simd_len; ++ne) {
            rem(simd_elem_index, ne, b[impl::bucket_index(e, 0)]);
          }
        });
    });
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
