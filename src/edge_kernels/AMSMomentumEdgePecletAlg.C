// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/AMSMomentumEdgePecletAlg.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <Enums.h>
#include <string>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpTypes.h>
#include <EquationSystem.h>
#include <ngp_utils/NgpFieldUtils.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <utils/AMSUtils.h>
#include <EigenDecomposition.h>

namespace sierra {
namespace nalu {

AMSMomentumEdgePecletAlg::AMSMomentumEdgePecletAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : Algorithm(realm, part),
    pecletNumber_(get_field_ordinal(
      realm.meta_data(), "peclet_number", stk::topology::EDGE_RANK)),
    pecletFactor_(get_field_ordinal(
      realm.meta_data(), "peclet_factor", stk::topology::EDGE_RANK)),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    tvisc_(get_field_ordinal(realm.meta_data(), "turbulent_viscosity")),
    coordinates_(get_field_ordinal(
      realm.meta_data(), realm.solutionOptions_->get_coordinates_name())),
    vrtm_(get_field_ordinal(
      realm.meta_data(), realm.does_mesh_move() ? "velocity_rtm" : "velocity")),
    edgeAreaVec_(get_field_ordinal(
      realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    pecScale_(realm.get_turb_model_constant(TM_ams_peclet_scale)),
    nDim_(realm.meta_data().spatial_dimension())
{
  const std::string dofName = "velocity";
  pecletFunction_ = eqSystem->ngp_create_peclet_function<double>(dofName);
}

void
AMSMomentumEdgePecletAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meta = realm_.meta_data();
  const auto ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  const auto density = fieldMgr.get_field<double>(density_);
  const auto viscosity = fieldMgr.get_field<double>(viscosity_);
  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vrtm = fieldMgr.get_field<double>(vrtm_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  auto pecletNumber = fieldMgr.get_field<double>(pecletNumber_);
  auto pecletFactor = fieldMgr.get_field<double>(pecletFactor_);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  // pointer for device capture
  auto* pecFunc = pecletFunction_;
  const int ndim = nDim_;
  const auto eps = eps_;

  nalu_ngp::run_edge_algorithm(
    "compute_peclet_factor", ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      NALU_ALIGNED DblType av[nalu_ngp::NDimMax];
      DblType asq{0.0}, axdx{0.0}, udotx{0.0};

      const auto edge = eInfo.meshIdx;
      for (int d = 0; d < ndim; d++) {
        av[d] = edgeAreaVec.get(edge, d);
      }
      const auto& nodes = eInfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);

      const DblType muIp =
        0.5 * (viscosity.get(nodeL, 0) + viscosity.get(nodeR, 0));
      const DblType mutIp = 0.5 * (tvisc.get(nodeL, 0) + tvisc.get(nodeR, 0));
      const DblType rhoIp =
        0.5 * (density.get(nodeL, 0) + density.get(nodeR, 0));

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
        udotx += 0.5 * dxj * (vrtm.get(nodeR, d) + vrtm.get(nodeL, d));
      }

      const DblType pecnum =
        rhoIp * stk::math::abs(udotx) / (muIp + pecScale_ * mutIp + eps);
      pecletNumber.get(edge, 0) = pecnum;
      pecletFactor.get(edge, 0) = pecFunc->execute(pecnum);
    });
}

} // namespace nalu
} // namespace sierra
