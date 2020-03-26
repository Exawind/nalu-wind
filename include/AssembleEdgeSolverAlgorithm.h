// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ASSEMBLEEDGESOLVERALGORITHM_H
#define ASSEMBLEEDGESOLVERALGORITHM_H

#include "SolverAlgorithm.h"
#include "ElemDataRequests.h"
#include "ElemDataRequestsGPU.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SharedMemData.h"
#include "EquationSystem.h"
#include "LinearSystem.h"

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra {
namespace nalu {

class AssembleEdgeSolverAlgorithm : public SolverAlgorithm
{
public:
  using DblType = double;
  using ShmemDataType = SharedMemData_Edge<DeviceTeamHandleType, DeviceShmem>;

  AssembleEdgeSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem);

  AssembleEdgeSolverAlgorithm() = delete;
  AssembleEdgeSolverAlgorithm(const AssembleEdgeSolverAlgorithm&) = delete;

  virtual ~AssembleEdgeSolverAlgorithm() = default;

  virtual void initialize_connectivity();

  template<typename LambdaFunction>
  void run_algorithm(stk::mesh::BulkData& bulk, LambdaFunction lambdaFunc)
  {
    const auto& meta = bulk.mesh_meta_data();
    const auto& ngpMesh = realm_.ngp_mesh();

    const int bytes_per_team = 0;
    const int bytes_per_thread = calc_shmem_bytes_per_thread_edge(rhsSize_);

    stk::mesh::Selector sel = meta.locally_owned_part() &
                              stk::mesh::selectUnion(partVec_) &
                              !(realm_.get_inactive_selector());

    const auto& buckets = ngp::get_bucket_ids(bulk, entityRank_, sel);
    auto team_exec = get_device_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

    // Create local copies of class data for device capture
    const auto entityRank = entityRank_;
    const auto rhsSize = rhsSize_;

    auto coeffApplier = coeff_applier();

    const auto nodesPerEntity = nodesPerEntity_;

    Kokkos::parallel_for(
      team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
        auto bktId = buckets.device_get(team.league_rank());
        auto& b = ngpMesh.get_bucket(entityRank, bktId);

        ShmemDataType smdata(team, rhsSize);

        const size_t bktLen = b.size();
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, bktLen),
          [&](const size_t& bktIndex) {
            auto edge = b[bktIndex];
            const auto edgeIndex = ngpMesh.fast_mesh_index(edge);
            smdata.ngpElemNodes = ngpMesh.get_nodes(entityRank, edgeIndex);

            const auto nodeL = ngpMesh.fast_mesh_index(smdata.ngpElemNodes[0]);
            const auto nodeR = ngpMesh.fast_mesh_index(smdata.ngpElemNodes[1]);

            set_vals(smdata.rhs, 0.0);
            set_vals(smdata.lhs, 0.0);

            lambdaFunc(smdata, edgeIndex, nodeL, nodeR);

            coeffApplier(
              nodesPerEntity, smdata.ngpElemNodes, smdata.scratchIds,
              smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
          });
      });
  }

protected:
  ElemDataRequests dataNeeded_;

  static constexpr stk::mesh::EntityRank entityRank_{stk::topology::EDGE_RANK};
  static constexpr int nodesPerEntity_{2};
  static constexpr int NDimMax_{3};
  const int rhsSize_;
};

}  // nalu
}  // sierra


#endif /* ASSEMBLEEDGESOLVERALGORITHM_H */
