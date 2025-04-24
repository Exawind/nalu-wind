// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "FixPressureAtNodeAlgorithm.h"
#include "FixPressureAtNodeInfo.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "SolutionOptions.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include "stk_mesh/base/NgpMesh.hpp"
#include <stk_util/parallel/ParallelReduce.hpp>

#include <limits>

namespace sierra {
namespace nalu {

FixPressureAtNodeAlgorithm::FixPressureAtNodeAlgorithm(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    info_(*(realm.solutionOptions_->fixPressureInfo_)),
    meshMotion_(realm.does_mesh_move())
{
  auto& meta = realm_.meta_data();

  coordinates_ = meta.get_field<double>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta.get_field<double>(stk::topology::NODE_RANK, "pressure");
}

FixPressureAtNodeAlgorithm::~FixPressureAtNodeAlgorithm() {}

void
FixPressureAtNodeAlgorithm::initialize_connectivity()
{
  /* Hypre GPU Assembly requires initialize to happen here (for graph creation),
   * not in execute */
  if (doInit_)
    initialize();
  eqSystem_->linsys_->buildDirichletNodeGraph(refNodeList_);
}

void
FixPressureAtNodeAlgorithm::execute()
{

  int numNodes = refNodeList_.size();
  STK_ThrowAssertMsg(
    numNodes <= 1,
    "Invalid number of nodes encountered in FixPressureAtNodeAlgorithm");

  stk::mesh::Entity targetNode = targetNode_;
  if (numNodes == 0 || !realm_.bulk_data().is_valid(targetNode)) {
    return;
  }

  // Reset LHS and RHS for this matrix
  CoeffApplier* deviceCoeffApplier = eqSystem_->linsys_->get_coeff_applier();

  stk::mesh::NgpMesh ngpMesh = realm_.ngp_mesh();
  NGPDoubleFieldType ngpPressure = realm_.ngp_field_manager().get_field<double>(
    pressure_->mesh_meta_data_ordinal());
  double refPressure = info_.refPressure_;
  const bool fixPressureNode = fixPressureNode_;

  const int bytes_per_team = 0;
  const int rhsSize = 1;
  const int bytes_per_thread =
    rhsSize * ((rhsSize + 1) * sizeof(double) + 2 * sizeof(int)) +
    sizeof(SharedMemView<double**, DeviceShmem>) +
    sizeof(SharedMemView<double*, DeviceShmem>) +
    2 * sizeof(SharedMemView<int*, DeviceShmem>);

  const int threads_per_team = 1;
  auto team_exec = get_device_team_policy(
    1, bytes_per_team, bytes_per_thread, threads_per_team);

  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
      auto lhs = get_shmem_view_2D<double, DeviceTeamHandleType, DeviceShmem>(
        team, rhsSize, rhsSize);
      auto rhs = get_shmem_view_1D<double, DeviceTeamHandleType, DeviceShmem>(
        team, rhsSize);
      auto scratchIds =
        get_shmem_view_1D<int, DeviceTeamHandleType, DeviceShmem>(
          team, rhsSize);
      auto sortPerm = get_shmem_view_1D<int, DeviceTeamHandleType, DeviceShmem>(
        team, rhsSize);

      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, 1), [=](const size_t&) {
          stk::mesh::NgpMesh::ConnectedNodes refNodeList(&targetNode, 1);
          deviceCoeffApplier->resetRows(1, &targetNode, 0, 1);

          // Fix the pressure for this node only if this is proc is owner
          if (numNodes > 0 && fixPressureNode) {
            sortPerm(0) = 0;
            const double pressureN = ngpPressure.get(ngpMesh, targetNode, 0);

            lhs(0, 0) = 1.0; // Set diagonal entry to 1.0
            rhs(0) = refPressure - pressureN;

            (*deviceCoeffApplier)(
              refNodeList.size(), refNodeList, scratchIds, sortPerm, rhs, lhs,
              __FILE__);
          }
        });
    });

  eqSystem_->linsys_->free_coeff_applier(deviceCoeffApplier);
}

void
FixPressureAtNodeAlgorithm::initialize()
{
  if (info_.lookupType_ == FixPressureAtNodeInfo::SPATIAL_LOCATION) {
    // Determine the nearest node where pressure is referenced
    auto nodeID = determine_nearest_node();
    process_pressure_fix_node(nodeID);
  } else if (info_.lookupType_ == FixPressureAtNodeInfo::STK_NODE_ID) {
    process_pressure_fix_node(info_.stkNodeId_);
  }

  // Flip init flag
  doInit_ = false;
}

stk::mesh::EntityId
FixPressureAtNodeAlgorithm::determine_nearest_node()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  int nDim = meta.spatial_dimension();
  auto& refLoc = info_.location_;

  // Get the target search part vector
  auto& partNames = info_.searchParts_;
  auto nParts = partNames.size();
  stk::mesh::PartVector parts(nParts);
  for (size_t i = 0; i < nParts; i++) {
    stk::mesh::Part* part = meta.get_part(partNames[i]);
    if (nullptr != part)
      parts[i] = part;
    else
      throw std::runtime_error(
        "FixPressureAtNodeAlgorithm: Target search part is null " +
        partNames[i]);
  }

  // Determine the nearest node in this processor
  stk::mesh::Entity nearestNode;
  stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(parts);
  auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, sel);

  double distSqr = std::numeric_limits<double>::max();
  for (auto b : buckets) {
    auto length = b->size();

    for (size_t i = 0; i < length; i++) {
      auto node = (*b)[i];
      const double* coords = stk::mesh::field_data(*coordinates_, node);

      double dist = 0.0;
      for (int j = 0; j < nDim; j++) {
        double xdiff = (refLoc[j] - coords[j]);
        dist += xdiff * xdiff;
      }
      if (dist < distSqr) {
        distSqr = dist;
        nearestNode = node;
      }
    }
  }

  // Determine the global minimum
  std::vector<double> minDistList(bulk.parallel_size());
  MPI_Allgather(
    &distSqr, 1, MPI_DOUBLE, minDistList.data(), 1, MPI_DOUBLE,
    bulk.parallel());
  int minDistProc = -1;
  double minDist = std::numeric_limits<double>::max();
  for (int i = 0; i < bulk.parallel_size(); i++) {
    if (minDistList[i] < minDist) {
      minDist = minDistList[i];
      minDistProc = i;
    }
  }

  // Communicate the nearest node ID to all processors.
  stk::mesh::EntityId nodeID = 0;
  stk::mesh::EntityId g_nodeID;
  if (minDistProc == bulk.parallel_rank())
    nodeID = bulk.identifier(nearestNode);
  stk::all_reduce_max(bulk.parallel(), &nodeID, &g_nodeID, 1);

  return g_nodeID;
}

void
FixPressureAtNodeAlgorithm::process_pressure_fix_node(
  const stk::mesh::EntityId nodeID)
{
  auto& bulk = realm_.bulk_data();

  // Store the target node on the owning processor as well as the shared
  // processors.
  targetNode_ = bulk.get_entity(stk::topology::NODE_RANK, nodeID);

  // If this node isn't on this MPI rank, return early
  if (!bulk.is_valid(targetNode_)) {
    fixPressureNode_ = false;
    return;
  }

  // For periodic simulations, make sure that the node doesn't lie on periodic
  // boundaries.
  if (realm_.periodic_mapping_) {
    stk::mesh::Selector pSel = realm_.periodic_mapping_->selector_a |
                               realm_.periodic_mapping_->selector_b;
    if (pSel(bulk.bucket(targetNode_))) {
      throw std::runtime_error(
        "FixPressureAtNode: Target node lies on a periodic boundary. This is "
        "not supported. Please change the target location.");
    }
  }

  if (bulk.bucket(targetNode_).owned() || bulk.bucket(targetNode_).shared()) {
    refNodeList_ = stk::mesh::NgpMesh::ConnectedNodes(&targetNode_, 1);

    // Only apply pressure correction on the owning processor
    fixPressureNode_ = bulk.bucket(targetNode_).owned();
  }
}

} // namespace nalu
} // namespace sierra
