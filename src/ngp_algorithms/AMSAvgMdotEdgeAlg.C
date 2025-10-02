// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/AMSAvgMdotEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

AMSAvgMdotEdgeAlg::AMSAvgMdotEdgeAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    avgVelocityRTM_(get_field_ordinal(
      realm.meta_data(),
      realm.does_mesh_move() ? "average_velocity_rtm" : "average_velocity")),
    densityNp1_(
      get_field_ordinal(realm.meta_data(), "density", stk::mesh::StateNP1)),
    edgeAreaVec_(get_field_ordinal(
      realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    avgMassFlowRate_(get_field_ordinal(
      realm.meta_data(), "average_mass_flow_rate", stk::topology::EDGE_RANK))
{
}

void
AMSAvgMdotEdgeAlg::execute()
{
  constexpr int NDimMax = 3;
  const auto& meta = realm_.meta_data();
  const int ndim = meta.spatial_dimension();

  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  // Interpolation option for rho*U
  const DblType interpTogether = realm_.get_mdot_interp();
  const DblType om_interpTogether = (1.0 - interpTogether);

  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto avgVelocity = fieldMgr.get_field<double>(avgVelocityRTM_);
  const auto density = fieldMgr.get_field<double>(densityNp1_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);

  auto avgMdot = fieldMgr.get_field<double>(avgMassFlowRate_);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    "compute_avgMdot_edge_interior", ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& einfo) {
       DblType av[NDimMax];
      const auto& nodes = einfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);

      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);

      const DblType rhoIp = 0.5 * (densityL + densityR);

      DblType tmdot = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType rhoUjIp = 0.5 * (densityR * avgVelocity.get(nodeR, d) +
                                       densityL * avgVelocity.get(nodeL, d));
        const DblType ujIp =
          0.5 * (avgVelocity.get(nodeR, d) + avgVelocity.get(nodeL, d));
        tmdot +=
          (interpTogether * rhoUjIp + om_interpTogether * rhoIp * ujIp) * av[d];
      }

      // Update edge field
      avgMdot.get(einfo.meshIdx, 0) = tmdot;
    });

  // Flag that the field has been modified on device for future sync
  avgMdot.modify_on_device();
}

} // namespace nalu
} // namespace sierra
