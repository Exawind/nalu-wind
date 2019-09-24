/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/TAMSAvgMdotEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

TAMSAvgMdotEdgeAlg::TAMSAvgMdotEdgeAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    avgTime_(get_field_ordinal(realm.meta_data(), "rans_time_scale")),
    massFlowRate_(get_field_ordinal(
      realm.meta_data(), "mass_flow_rate", stk::topology::EDGE_RANK)),
    avgMassFlowRate_(get_field_ordinal(
      realm.meta_data(), "average_mass_flow_rate", stk::topology::EDGE_RANK))
{
}

void
TAMSAvgMdotEdgeAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<ngp::Mesh>;
  const DblType dt = realm_.get_time_step();
  const auto& meta = realm_.meta_data();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  const auto avgTime = fieldMgr.get_field<double>(avgTime_);
  const auto mdot = fieldMgr.get_field<double>(massFlowRate_);
  auto avgMdot = fieldMgr.get_field<double>(avgMassFlowRate_);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    ngpMesh, sel, KOKKOS_LAMBDA(const EntityInfoType& einfo) {
      const auto& nodes = einfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);

      const DblType avgTimeL = avgTime.get(nodeL, 0);
      const DblType avgTimeR = avgTime.get(nodeR, 0);

      const DblType avgTimeIp = 0.5 * (avgTimeR + avgTimeL);

      const DblType weightAvg = std::max(1.0 - dt / avgTimeIp, 0.0);
      const DblType weightInst = std::min(dt / avgTimeIp, 1.0);

      avgMdot.get(einfo.meshIdx, 0) =
        weightAvg * avgMdot.get(einfo.meshIdx, 0) +
        weightInst * mdot.get(einfo.meshIdx, 0);
    });

  // Flag that the field has been modified on device for future sync
  avgMdot.modify_on_device();
}

} // namespace nalu
} // namespace sierra
