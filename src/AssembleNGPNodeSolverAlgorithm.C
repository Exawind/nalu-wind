// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "AssembleNGPNodeSolverAlgorithm.h"
#include "EquationSystem.h"
#include "KokkosInterface.h"
#include "LinearSystem.h"
#include "Realm.h"

#include "node_kernels/NodeKernel.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

namespace {
inline int
calc_shmem_bytes_per_thread(int rhsSize)
{
  // LHS (RHS^2) + RHS
  const int matSize = rhsSize * (1 + rhsSize) * sizeof(double);
  // Scratch IDs and search permutations (will be optimized later)
  const int idSize = 2 * rhsSize * sizeof(int);

  return (matSize + idSize);
}

template <typename TEAMHANDLETYPE, typename SHMEM>
struct SharedMemData_Node
{
  KOKKOS_FUNCTION
  SharedMemData_Node(const TEAMHANDLETYPE& team, unsigned rhsSize)
    : ngpNodes(nodeID, 1)
  {
    rhs = get_shmem_view_1D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    lhs =
      get_shmem_view_2D<double, TEAMHANDLETYPE, SHMEM>(team, rhsSize, rhsSize);
    scratchIds = get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
    sortPermutation =
      get_shmem_view_1D<int, TEAMHANDLETYPE, SHMEM>(team, rhsSize);
  }

  stk::mesh::Entity nodeID[1];
  stk::mesh::NgpMesh::ConnectedNodes ngpNodes;
  SharedMemView<double*, SHMEM> rhs;
  SharedMemView<double**, SHMEM> lhs;

  SharedMemView<int*, SHMEM> scratchIds;
  SharedMemView<int*, SHMEM> sortPermutation;
};
} // namespace

AssembleNGPNodeSolverAlgorithm::AssembleNGPNodeSolverAlgorithm(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    rhsSize_(eqSystem->linsys_->numDof())
{
}

AssembleNGPNodeSolverAlgorithm::~AssembleNGPNodeSolverAlgorithm()
{
  // Release the device pointers if any
  for (auto& kern : nodeKernels_) {
    kern->free_on_device();
  }
}

void
AssembleNGPNodeSolverAlgorithm::initialize_connectivity()
{
  const size_t numKernels = nodeKernels_.size();
  if (numKernels < 1)
    return;

  eqSystem_->linsys_->buildNodeGraph(partVec_);
}

void
AssembleNGPNodeSolverAlgorithm::execute()
{
  using ShmemDataType = SharedMemData_Node<DeviceTeamHandleType, DeviceShmem>;

  const size_t numKernels = nodeKernels_.size();
  if (numKernels < 1)
    return;

  for (auto& kern : nodeKernels_)
    kern->setup(realm_);

  auto ngpKernels = nalu_ngp::create_ngp_view<NodeKernel>(nodeKernels_);
  auto coeffApplier = coeff_applier();

  const auto& meta = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;
  const int rhsSize = rhsSize_;

  const int nodesPerEntity = 1;
  const int bytes_per_team = 0;
  const int bytes_per_thread = calc_shmem_bytes_per_thread(rhsSize);

  stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_) &
    !(realm_.replicated_periodic_node_selector()) &
    !(realm_.get_inactive_selector());
  const auto& buckets =
    stk::mesh::get_bucket_ids(realm_.bulk_data(), entityRank, sel);

  auto team_exec =
    get_device_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& b = ngpMesh.get_bucket(entityRank, bktId);

      ShmemDataType smdata(team, rhsSize);

      const size_t bktLen = b.size();
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bktLen), [&](const size_t& bktIndex) {
          auto node = b[bktIndex];
          const auto nodeIndex = ngpMesh.fast_mesh_index(node);
          smdata.nodeID[0] = node;

          set_vals(smdata.rhs, 0.0);
          set_vals(smdata.lhs, 0.0);

          for (size_t i = 0; i < numKernels; ++i) {
            NodeKernel* kernel = ngpKernels(i);
            kernel->execute(smdata.lhs, smdata.rhs, nodeIndex);
          }

          coeffApplier(
            nodesPerEntity, smdata.ngpNodes, smdata.scratchIds,
            smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
        });
    });
  coeffApplier.free_coeff_applier();
}

} // namespace nalu
} // namespace sierra
