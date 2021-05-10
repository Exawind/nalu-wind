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
    beta_(get_field_ordinal(realm.meta_data(), "k_ratio")),
    tke_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    sdr_(get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    nodalMij_(get_field_ordinal(realm.meta_data(), "metric_tensor")),
    avgResAdeq_(
      get_field_ordinal(realm.meta_data(), "avg_res_adequacy_parameter")),
    coordinates_(get_field_ordinal(
      realm.meta_data(), realm.solutionOptions_->get_coordinates_name())),
    vrtm_(get_field_ordinal(
      realm.meta_data(), realm.does_mesh_move() ? "velocity_rtm" : "velocity")),
    edgeAreaVec_(get_field_ordinal(
      realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg)),
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
  const auto beta = fieldMgr.get_field<double>(beta_);
  const auto tke = fieldMgr.get_field<double>(tke_);
  const auto sdr = fieldMgr.get_field<double>(sdr_);
  const auto avgResAdeq = fieldMgr.get_field<double>(avgResAdeq_);
  const auto nodalMij = fieldMgr.get_field<double>(nodalMij_);
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
      const DblType tkeIp = 0.5 * (stk::math::max(tke.get(nodeL, 0), 1.0e-12) +
                                   stk::math::max(tke.get(nodeR, 0), 1.0e-12));
      const DblType sdrIp = 0.5 * (stk::math::max(sdr.get(nodeL, 0), 1.0e-12) +
                                   stk::math::max(sdr.get(nodeR, 0), 1.0e-12));
      const DblType alphaIp = 0.5 * (stk::math::pow(beta.get(nodeL, 0), 1.7) +
                                     stk::math::pow(beta.get(nodeR, 0), 1.7));
      const DblType epsilon13Ip =
        stk::math::pow(betaStar_ * tkeIp * sdrIp, 1.0 / 3.0);

      const DblType muRansIp = alphaIp * (2.0 - alphaIp) * mutIp;

      // Mij, eigenvectors and eigenvalues
      DblType Mij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType Q[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType D[nalu_ngp::NDimMax][nalu_ngp::NDimMax];

      for (int i = 0; i < ndim; i++)
        for (int j = 0; j < ndim; j++)
          Mij[i][j] = 0.5 * (nodalMij.get(nodeL, i * ndim + j) +
                             nodalMij.get(nodeR, i * ndim + j));

      EigenDecomposition::sym_diagonalize<DblType>(Mij, Q, D);

      // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
      // so to create M43, we use Q D^(4/3) Q^T
      DblType M43[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < ndim; i++)
        for (int j = 0; j < ndim; j++)
          M43[i][j] = 0.0;

      const double fourThirds = 4. / 3.;
      for (int k = 0; k < ndim; k++) {
        const DblType D43 = stk::math::pow(D[k][k], fourThirds);
        for (int i = 0; i < ndim; i++) {
          for (int j = 0; j < ndim; j++) {
            M43[i][j] += Q[i][k] * Q[j][k] * D43;
          }
        }
      }
      DblType traceM43 = 0.0;
      for (int i = 0; i < ndim; i++) {
        traceM43 += M43[i][i];
      }

      // Compute CM43
      DblType CM43 =
        ams_utils::get_M43_constant<DblType, nalu_ngp::NDimMax>(D, CMdeg_);

      const DblType CM43scale = stk::math::max(
        stk::math::min(
          0.5 * (stk::math::pow(avgResAdeq.get(nodeL, 0), 2.0) +
                 stk::math::pow(avgResAdeq.get(nodeR, 0), 2.0)),
          30.0),
        1.0);

      const DblType muM43Ip = 0.5 * CM43scale * CM43 * epsilon13Ip * traceM43;

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
        udotx += 0.5 * dxj * (vrtm.get(nodeR, d) + vrtm.get(nodeL, d));
      }

      const DblType pecnum =
        rhoIp * stk::math::abs(udotx) / (muIp + muRansIp + muM43Ip + eps);
      pecletNumber.get(edge, 0) = pecnum;
      pecletFactor.get(edge, 0) = pecFunc->execute(pecnum);
    });
}

} // namespace nalu
} // namespace sierra
