// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/SSTAMSAveragesAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "EigenDecomposition.h"
#include "utils/AMSUtils.h"
#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

SSTAMSAveragesAlg::SSTAMSAveragesAlg(Realm& realm, stk::mesh::Part* part)
  : AMSAveragesAlg(realm, part),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg)),
    v2cMu_(realm.get_turb_model_constant(TM_v2cMu)),
    aspectRatioSwitch_(realm.get_turb_model_constant(TM_aspRatSwitch)),
    avgTimeCoeff_(realm.get_turb_model_constant(TM_avgTimeCoeff)),
    alphaPow_(realm.get_turb_model_constant(TM_alphaPow)),
    alphaScaPow_(realm.get_turb_model_constant(TM_alphaScaPow)),
    coeffR_(realm.get_turb_model_constant(TM_coeffR)),
    meshMotion_(realm.does_mesh_move()),
    RANSBelowKs_(realm_.solutionOptions_->RANSBelowKs_),
    z0_(realm_.solutionOptions_->roughnessHeight_),
    lengthScaleLimiter_(realm_.solutionOptions_->lengthScaleLimiter_),
    eastVector_(realm_.solutionOptions_->eastVector_),
    northVector_(realm_.solutionOptions_->northVector_),
    velocity_(get_field_ordinal(realm.meta_data(), "velocity")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    dudx_(get_field_ordinal(realm.meta_data(), "dudx")),
    resAdeq_(
      get_field_ordinal(realm.meta_data(), "resolution_adequacy_parameter")),
    turbKineticEnergy_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    specDissipationRate_(
      get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    avgVelocity_(get_field_ordinal(realm.meta_data(), "average_velocity")),
    avgVelocityN_(get_field_ordinal(
      realm.meta_data(), "average_velocity", stk::mesh::StateN)),
    avgDudx_(get_field_ordinal(realm.meta_data(), "average_dudx")),
    avgDudxN_(
      get_field_ordinal(realm.meta_data(), "average_dudx", stk::mesh::StateN)),
    avgTkeRes_(get_field_ordinal(realm.meta_data(), "average_tke_resolved")),
    avgTkeResN_(get_field_ordinal(
      realm.meta_data(), "average_tke_resolved", stk::mesh::StateN)),
    avgProd_(get_field_ordinal(realm.meta_data(), "average_production")),
    avgProdN_(get_field_ordinal(
      realm.meta_data(), "average_production", stk::mesh::StateN)),
    avgTime_(get_field_ordinal(realm.meta_data(), "rans_time_scale")),
    avgResAdeq_(
      get_field_ordinal(realm.meta_data(), "avg_res_adequacy_parameter")),
    avgResAdeqN_(get_field_ordinal(
      realm.meta_data(), "avg_res_adequacy_parameter", stk::mesh::StateN)),
    tvisc_(get_field_ordinal(realm.meta_data(), "turbulent_viscosity")),
    visc_(get_field_ordinal(realm.meta_data(), "viscosity")),
    beta_(get_field_ordinal(realm.meta_data(), "k_ratio")),
    Mij_(get_field_ordinal(realm.meta_data(), "metric_tensor")),
    wallDist_(get_field_ordinal(realm.meta_data(), "minimum_distance_to_wall")),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name()))
{
}

void
SSTAMSAveragesAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();
  if (meta.spatial_dimension() != 3) {
    throw std::runtime_error("SSTAMSAveragesAlg only supported in 3D.");
  }
  const DblType dt = realm_.get_time_step();

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(
      *meta.get_field(stk::topology::NODE_RANK, "average_velocity"));

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  const auto visc = fieldMgr.get_field<double>(visc_);
  const auto tke = fieldMgr.get_field<double>(turbKineticEnergy_);
  const auto sdr = fieldMgr.get_field<double>(specDissipationRate_);
  const auto density = fieldMgr.get_field<double>(density_);
  auto beta = fieldMgr.get_field<double>(beta_);
  auto avgProd = fieldMgr.get_field<double>(avgProd_);
  auto avgProdN = fieldMgr.get_field<double>(avgProdN_);
  auto avgTkeRes = fieldMgr.get_field<double>(avgTkeRes_);
  auto avgTkeResN = fieldMgr.get_field<double>(avgTkeResN_);
  auto avgTime = fieldMgr.get_field<double>(avgTime_);
  auto resAdeq = fieldMgr.get_field<double>(resAdeq_);
  auto avgResAdeq = fieldMgr.get_field<double>(avgResAdeq_);
  auto avgResAdeqN = fieldMgr.get_field<double>(avgResAdeqN_);
  const auto vel = fieldMgr.get_field<double>(velocity_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  auto avgVel = fieldMgr.get_field<double>(avgVelocity_);
  auto avgVelN = fieldMgr.get_field<double>(avgVelocityN_);
  auto avgDudx = fieldMgr.get_field<double>(avgDudx_);
  auto avgDudxN = fieldMgr.get_field<double>(avgDudxN_);
  const auto Mij = fieldMgr.get_field<double>(Mij_);
  const auto wallDist = fieldMgr.get_field<double>(wallDist_);
  const auto coords = fieldMgr.get_field<double>(coordinates_);

  const DblType betaStar = betaStar_;
  const DblType CMdeg = CMdeg_;
  const DblType v2cMu = v2cMu_;
  const DblType beta_kol_local = beta_kol;
  const DblType aspectRatioSwitch = aspectRatioSwitch_;
  const DblType avgTimeCoeff = avgTimeCoeff_;
  const auto lengthScaleLimiter = lengthScaleLimiter_;
  const DblType alphaPow = alphaPow_;
  const DblType alphaScaPow = alphaScaPow_;
  const DblType coeffR = coeffR_;

  const bool RANSBelowKs = RANSBelowKs_;
  DblType k_s = 0;
  int gravity_i = 0;
  if (RANSBelowKs) {
    // relationship b/w sand grain roughness height, k_s, and aerodynamic
    // roughness, z0, as described in ref. Bau11, Eq. (2.29)
    k_s = 30. * z0_;
    for (int i = 0; i < 3; ++i) {
      if ((eastVector_[i] == 0.0) && (northVector_[i] == 0.0)) {
        gravity_i = i;
      }
    }
  }

  nalu_ngp::run_entity_algorithm(
    "SSTAMSAveragesAlg_computeAverages", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      // Calculate alpha
      if (tke.get(mi, 0) == 0.0)
        beta.get(mi, 0) = 1.0;
      else if ((RANSBelowKs) && (coords.get(mi, gravity_i) <= k_s)) {
        beta.get(mi, 0) = 1.0;
      } else {
        beta.get(mi, 0) =
          (tke.get(mi, 0) - avgTkeRes.get(mi, 0)) / tke.get(mi, 0);

        // limiters
        beta.get(mi, 0) = stk::math::min(beta.get(mi, 0), 1.0);

        beta.get(mi, 0) = stk::math::max(beta.get(mi, 0), beta_kol_local);
      }

      const DblType alpha = stk::math::pow(beta.get(mi, 0), alphaPow);

      // store RANS time scale
      if (lengthScaleLimiter) {
        const DblType l_t = stk::math::sqrt(tke.get(mi, 0)) /
                            (stk::math::pow(betaStar, .25) * sdr.get(mi, 0));
        avgTime.get(mi, 0) =
          avgTimeCoeff * l_t / stk::math::sqrt(tke.get(mi, 0));
      } else {
        avgTime.get(mi, 0) = avgTimeCoeff / (betaStar * sdr.get(mi, 0));
      }

      // causal time average ODE: d<phi>/dt = 1/avgTime * (phi - <phi>)
      const DblType weightAvg =
        stk::math::max(1.0 - dt / avgTime.get(mi, 0), 0.0);
      const DblType weightInst = stk::math::min(dt / avgTime.get(mi, 0), 1.0);

      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        avgVel.get(mi, i) =
          weightAvg * avgVelN.get(mi, i) + weightInst * vel.get(mi, i);

      DblType tkeRes = 0.0;
      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        tkeRes += (vel.get(mi, i) - avgVel.get(mi, i)) *
                  (vel.get(mi, i) - avgVel.get(mi, i));

      avgTkeRes.get(mi, 0) =
        weightAvg * avgTkeResN.get(mi, 0) + weightInst * 0.5 * tkeRes;

      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          avgDudx.get(mi, i * nalu_ngp::NDimMax + j) =
            weightAvg * avgDudxN.get(mi, i * nalu_ngp::NDimMax + j) +
            weightInst * dudx.get(mi, i * nalu_ngp::NDimMax + j);
        }
      }

      // Production averaging
      DblType tij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          const DblType avgSij =
            0.5 * (avgDudx.get(mi, i * nalu_ngp::NDimMax + j) +
                   avgDudx.get(mi, j * nalu_ngp::NDimMax + i));
          tij[i][j] = 2.0 * stk::math::pow(alpha, alphaScaPow) * (2.0 - alpha) *
                      tvisc.get(mi, 0) * avgSij;
        }
      }

      DblType Pij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          Pij[i][j] = 0.0;
          for (int m = 0; m < nalu_ngp::NDimMax; ++m) {
            Pij[i][j] +=
              avgDudx.get(mi, i * nalu_ngp::NDimMax + m) * tij[j][m] +
              avgDudx.get(mi, j * nalu_ngp::NDimMax + m) * tij[i][m];
          }
          Pij[i][j] *= 0.5;
        }
      }

      DblType P_res = 0.0;
      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          P_res += density.get(mi, 0) *
                   avgDudx.get(mi, i * nalu_ngp::NDimMax + j) *
                   ((avgVel.get(mi, i) - vel.get(mi, i)) *
                    (avgVel.get(mi, j) - vel.get(mi, j)));
        }
      }

      DblType instProd = 0.0;
      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        instProd += Pij[i][i];

      instProd -= P_res;

      // Use a longer averaging timescale for production
      const DblType prodAvgTime = 4.0 * avgTime.get(mi, 0);

      // causal time average ODE: d<phi>/dt = 1/avgTime * (phi - <phi>)
      const DblType prodWeightAvg = stk::math::max(1.0 - dt / prodAvgTime, 0.0);
      const DblType prodWeightInst = stk::math::min(dt / prodAvgTime, 1.0);

      avgProd.get(mi, 0) =
        prodWeightAvg * avgProdN.get(mi, 0) + prodWeightInst * instProd;

      // get Mij field_data
      DblType p_Mij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType PM[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType Q[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType D[nalu_ngp::NDimMax][nalu_ngp::NDimMax];

      for (int i = 0; i < nalu_ngp::NDimMax; i++) {
        const int iNdim = i * nalu_ngp::NDimMax;
        for (int j = 0; j < nalu_ngp::NDimMax; j++) {
          p_Mij[i][j] = Mij.get(mi, iNdim + j);
        }
      }

      // Eigenvalue decomposition of metric tensor
      EigenDecomposition::sym_diagonalize<DblType>(p_Mij, Q, D);

      // initialize M43 to 0
      DblType M43[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        for (int j = 0; j < nalu_ngp::NDimMax; ++j)
          M43[i][j] = 0.0;

      const DblType fourThirds = 4.0 / 3.0;

      for (int l = 0; l < nalu_ngp::NDimMax; l++) {
        const DblType D43 = stk::math::pow(D[l][l], fourThirds);
        for (int i = 0; i < nalu_ngp::NDimMax; i++) {
          for (int j = 0; j < nalu_ngp::NDimMax; j++) {
            M43[i][j] += Q[i][l] * Q[j][l] * D43;
          }
        }
      }

      const DblType maxEigM =
        stk::math::max(D[0][0], stk::math::max(D[1][1], D[2][2]));
      const DblType minEigM =
        stk::math::min(D[0][0], stk::math::min(D[1][1], D[2][2]));

      const DblType aspectRatio = maxEigM / minEigM;

      // zeroing out tensors
      DblType tauSGRS[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType tauSGET[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType tau[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType Psgs[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          tauSGRS[i][j] = 0.0;
          tauSGET[i][j] = 0.0;
          tau[i][j] = 0.0;
          Psgs[i][j] = 0.0;
        }
      }

      const DblType CM43 =
        ams_utils::get_M43_constant<DblType, nalu_ngp::NDimMax>(D, CMdeg);

      const DblType CM43scale = stk::math::max(
        stk::math::min(stk::math::pow(avgResAdeq.get(mi, 0), 2.0), 30.0), 1.0);

      const DblType epsilon13 =
        stk::math::pow(betaStar * tke.get(mi, 0) * sdr.get(mi, 0), 1.0 / 3.0);

      const DblType arScale = stk::math::if_then_else(
        aspectRatio > aspectRatioSwitch,
        1.0 - stk::math::tanh((aspectRatio - aspectRatioSwitch) / 10.0), 1.0);

      const DblType arInvScale = 1.0 - arScale;

      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          // Calculate tauSGRS_ij = 2*alpha*nu_t*<S_ij> where nu_t comes from
          // the SST model and <S_ij> is the strain rate tensor based on the
          // mean quantities... i.e this is (tauSGRS = alpha*tauSST)
          // The 2 in the coeff cancels with the 1/2 in the strain rate tensor
          const DblType coeffSGRS = stk::math::pow(alpha, alphaScaPow) *
                                    (2.0 - alpha) * tvisc.get(mi, 0) /
                                    density.get(mi, 0);
          tauSGRS[i][j] = avgDudx.get(mi, i * nalu_ngp::NDimMax + j) +
                          avgDudx.get(mi, j * nalu_ngp::NDimMax + i);
          tauSGRS[i][j] *= coeffSGRS;

          for (int l = 0; l < nalu_ngp::NDimMax; ++l) {
            // Calculate tauSGET_ij = CM43*<eps>^(1/3)*(M43_ik*dkuj' +
            // M43_jkdkui') where <eps> is the mean dissipation backed out from
            // the SST mean k and mean omega and dkuj' is the fluctuating
            // velocity gradients.
            const DblType coeffSGET = CM43scale * CM43 * epsilon13;
            const DblType fluctDudx_jl =
              dudx.get(mi, j * nalu_ngp::NDimMax + l) -
              avgDudx.get(mi, j * nalu_ngp::NDimMax + l);
            const DblType fluctDudx_il =
              dudx.get(mi, i * nalu_ngp::NDimMax + l) -
              avgDudx.get(mi, i * nalu_ngp::NDimMax + l);
            tauSGET[i][j] +=
              coeffSGET * arScale *
              (M43[i][l] * fluctDudx_jl + M43[j][l] * fluctDudx_il);
          }
          tauSGET[i][j] += arInvScale * tvisc.get(mi, 0) / density.get(mi, 0) *
                           (dudx.get(mi, i * nalu_ngp::NDimMax + j) -
                            avgDudx.get(mi, i * nalu_ngp::NDimMax + j) +
                            dudx.get(mi, j * nalu_ngp::NDimMax + i) -
                            avgDudx.get(mi, j * nalu_ngp::NDimMax + i));
        }
      }

      // Remove trace of tauSGET
      DblType tauSGET_tr = 0.0;
      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        tauSGET_tr += tauSGET[i][i];

      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        tauSGET[i][i] -= tauSGET_tr / nalu_ngp::NDimMax;

      // Calculate the full subgrid stress including the isotropic portion
      for (int i = 0; i < nalu_ngp::NDimMax; ++i)
        for (int j = 0; j < nalu_ngp::NDimMax; ++j)
          tau[i][j] =
            tauSGRS[i][j] + tauSGET[i][j] -
            ((i == j) ? 2.0 / 3.0 * beta.get(mi, 0) * tke.get(mi, 0) : 0.0);

      // Calculate the SGS production PSGS_ij = 1/2(tau_ik*djuk + tau_jk*diuk)
      // where diuj is the instantaneous velocity gradients
      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          for (int l = 0; l < nalu_ngp::NDimMax; ++l) {
            Psgs[i][j] += tau[i][l] * dudx.get(mi, l * nalu_ngp::NDimMax + j) +
                          tau[j][l] * dudx.get(mi, l * nalu_ngp::NDimMax + i);
          }
          Psgs[i][j] *= 0.5;
        }
      }

      for (int i = 0; i < nalu_ngp::NDimMax; ++i) {
        for (int j = 0; j < nalu_ngp::NDimMax; ++j) {
          PM[i][j] = 0.0;
          for (int l = 0; l < nalu_ngp::NDimMax; ++l)
            PM[i][j] += Psgs[i][l] * p_Mij[l][j];
        }
      }

      // Scale PM first
      const DblType v2 =
        1.0 / v2cMu *
        (tvisc.get(mi, 0) / density.get(mi, 0) / avgTime.get(mi, 0));
      const DblType PMscale = coeffR * stk::math::pow(1.5 * beta.get(mi, 0) * v2, -1.5);

      // Handle case where tke = 0, should only occur at a wall boundary
      if (tke.get(mi, 0) == 0.0)
        resAdeq.get(mi, 0) = 1.0;
      else if ((RANSBelowKs) && (coords.get(mi, gravity_i) <= k_s)) {
        resAdeq.get(mi, 0) = 1.0;
      } else {
        for (int i = 0; i < nalu_ngp::NDimMax; ++i)
          for (int j = 0; j < nalu_ngp::NDimMax; ++j)
            PM[i][j] = PM[i][j] * PMscale;

        DblType PMmag = 0.0;
        for (int i = 0; i < nalu_ngp::NDimMax; ++i)
          for (int j = 0; j < nalu_ngp::NDimMax; ++j)
            PMmag += PM[i][j] * PM[i][j];

        PMmag = stk::math::sqrt(PMmag);

        EigenDecomposition::general_eigenvalues<DblType>(PM, Q, D);

        // Take only positive eigenvalues of PM
        DblType maxPM =
          stk::math::max(D[0][0], stk::math::max(D[1][1], D[2][2]));

        // Update the instantaneous resAdeq field
        resAdeq.get(mi, 0) =
          stk::math::max(stk::math::min(maxPM, PMmag), 0.00000001);

        resAdeq.get(mi, 0) = stk::math::min(resAdeq.get(mi, 0), 30.0);

        if (alpha >= 1.0) {
          resAdeq.get(mi, 0) = stk::math::min(resAdeq.get(mi, 0), 1.0);
        }

        if (alpha <= beta_kol_local)
          resAdeq.get(mi, 0) = stk::math::max(resAdeq.get(mi, 0), 1.0);
      }

      avgResAdeq.get(mi, 0) =
        weightAvg * avgResAdeqN.get(mi, 0) + weightInst * resAdeq.get(mi, 0);
    });
}

} // namespace nalu
} // namespace sierra
