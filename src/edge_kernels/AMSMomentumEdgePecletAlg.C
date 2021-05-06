// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/AMSMomentumPecletEdgeAlg.h>
#include <Realm.h>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpFieldUtils.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <SolutionOptions.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

AMSMomentumPecletEdgeAlg::AMSMomentumPecletEdgeAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    velocityName_("velocity"),
    pecletFactor_(get_field_ordinal(
      realm_.meta_data(), "peclet_factor", stk::topology::EDGE_RANK)),
    beta_(get_field_ordinal(realm.meta_data(), "k_ratio"))
{
  const DblType alpha = realm_.get_alpha_factor(velocityName_);
  const DblType alphaUpw = realm_.get_alpha_upw_factor(velocityName_);
  const DblType hoUpwind = realm_.get_upw_factor(velocityName_);
  // check that upwinding factors are set so we only blend with the peclet
  // parameter
  // treat as error for now, could switch to warning though.
  std::string error_message;
  if (alpha != 0.0)
    error_message += "alpha is set to: " + std::to_string(alpha) +
                     " alpha should be 0.0 when using AMS\n";
  if (alphaUpw != 1.0)
    error_message += "alpha_upw is set to: " + std::to_string(alphaUpw) +
                     " alpha_upw should be 1.0 when using AMS\n";
  if (hoUpwind != 1.0)
    error_message += "upw_factor is set to: " + std::to_string(hoUpwind) +
                     " upw_factor should be 1.0 when using AMS\n";

  if (!error_message.empty())
    NaluEnv::self().naluOutputP0() << "WARNING::For the momementum equation:\n"
                                   << error_message;
}

void
AMSMomentumPecletEdgeAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  stk::mesh::MetaData& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto pecFactor = fieldMgr.get_field<double>(pecletFactor_);
  const auto beta = fieldMgr.get_field<double>(beta_);

  const DblType ams_peclet_offset = realm_.get_turb_model_constant(TM_ams_peclet_offset);
  const DblType ams_peclet_slope = realm_.get_turb_model_constant(TM_ams_peclet_slope);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    "compute_ams_alpha_upw", ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      const auto& nodes = eInfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);
      const auto edge = eInfo.meshIdx;

      // compute edge quantities
      const DblType betaEdge = 0.5 * (beta.get(nodeL, 0) + beta.get(nodeR, 0));

      pecFactor.get(edge, 0) =
        0.5 * stk::math::tanh(
                ams_peclet_slope * (betaEdge - ams_peclet_offset) + 1.0);
    });
}

} // namespace nalu
} // namespace sierra
