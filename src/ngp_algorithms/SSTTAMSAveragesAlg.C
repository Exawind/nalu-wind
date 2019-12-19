// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "ngp_algorithms/SSTTAMSAveragesAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "EigenDecomposition.h"
#include "utils/TAMSUtils.h"

namespace sierra {
namespace nalu {

SSTTAMSAveragesAlg::SSTTAMSAveragesAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg)),
    v2cMu_(realm.get_turb_model_constant(TM_v2cMu)),
    meshMotion_(realm.does_mesh_move()),
    velocity_(get_field_ordinal(realm.meta_data(), "velocity")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    dudx_(get_field_ordinal(realm.meta_data(), "dudx")),
    resAdeq_(
      get_field_ordinal(realm.meta_data(), "resolution_adequacy_parameter")),
    turbKineticEnergy_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    specDissipationRate_(
      get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    avgVelocity_(get_field_ordinal(realm.meta_data(), "average_velocity")),
    avgDudx_(get_field_ordinal(realm.meta_data(), "average_dudx")),
    avgTkeRes_(get_field_ordinal(realm.meta_data(), "average_tke_resolved")),
    avgProd_(get_field_ordinal(realm.meta_data(), "average_production")),
    avgTime_(get_field_ordinal(realm.meta_data(), "rans_time_scale")),
    avgResAdeq_(
      get_field_ordinal(realm.meta_data(), "avg_res_adequacy_parameter")),
    tvisc_(get_field_ordinal(realm.meta_data(), "turbulent_viscosity")),
    alpha_(get_field_ordinal(realm.meta_data(), "k_ratio")),
    Mij_(get_field_ordinal(realm.meta_data(), "metric_tensor"))
{
}

void
SSTTAMSAveragesAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  const auto& meta = realm_.meta_data();
  const DblType dt = realm_.get_time_step();
  const int nDim = meta.spatial_dimension();

  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(
      *meta.get_field(stk::topology::NODE_RANK, "average_velocity"));

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const auto tvisc = fieldMgr.get_field<double>(tvisc_);
  const auto tke = fieldMgr.get_field<double>(turbKineticEnergy_);
  const auto sdr = fieldMgr.get_field<double>(specDissipationRate_);
  const auto density = fieldMgr.get_field<double>(density_);
  auto alpha = fieldMgr.get_field<double>(alpha_);
  auto avgProd = fieldMgr.get_field<double>(avgProd_);
  auto avgTkeRes = fieldMgr.get_field<double>(avgTkeRes_);
  auto avgTime = fieldMgr.get_field<double>(avgTime_);
  auto resAdeq = fieldMgr.get_field<double>(resAdeq_);
  auto avgResAdeq = fieldMgr.get_field<double>(avgResAdeq_);
  const auto vel = fieldMgr.get_field<double>(velocity_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  auto avgVel = fieldMgr.get_field<double>(avgVelocity_);
  auto avgDudx = fieldMgr.get_field<double>(avgDudx_);
  const auto Mij = fieldMgr.get_field<double>(Mij_);

  const DblType betaStar = betaStar_;
  const DblType CMdeg = CMdeg_;
  const DblType v2cMu = v2cMu_;
  const DblType alpha_kol_local = alpha_kol;

  nalu_ngp::run_entity_algorithm(
    "SSTTAMSAveragesAlg_computeAverages",
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      // Calculate alpha
      if (tke.get(mi, 0) == 0.0)
        alpha.get(mi, 0) = 1.0;
      else {
        alpha.get(mi, 0) = 1.0 - avgTkeRes.get(mi, 0) / tke.get(mi, 0);

        // limiters
        alpha.get(mi, 0) = stk::math::min(alpha.get(mi, 0), 1.0);

        alpha.get(mi, 0) = stk::math::max(alpha.get(mi, 0), alpha_kol_local);
      }

      // store RANS time scale
      avgTime.get(mi, 0) = 1.0 / (betaStar * sdr.get(mi, 0));

      // causal time average ODE: d<phi>/dt = 1/avgTime * (phi - <phi>)
      const DblType weightAvg = stk::math::max(1.0 - dt / avgTime.get(mi, 0), 0.0);
      const DblType weightInst = stk::math::min(dt / avgTime.get(mi, 0), 1.0);

      DblType tkeRes = 0.0;
      for (int i = 0; i < nDim; ++i)
        tkeRes += (vel.get(mi, i) - avgVel.get(mi, i)) *
                  (vel.get(mi, i) - avgVel.get(mi, i));
      avgTkeRes.get(mi, 0) =
        weightAvg * avgTkeRes.get(mi, 0) + weightInst * 0.5 * tkeRes;

      for (int i = 0; i < nDim; ++i)
        avgVel.get(mi, i) =
          weightAvg * avgVel.get(mi, i) + weightInst * vel.get(mi, i);

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          avgDudx.get(mi, i * nDim + j) =
            weightAvg * avgDudx.get(mi, i * nDim + j) +
            weightInst * dudx.get(mi, i * nDim + j);
        }
      }

      // Production averaging
      DblType tij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          const DblType avgSij = 0.5 * (avgDudx.get(mi, i * nDim + j) +
                                        avgDudx.get(mi, j * nDim + i));
          tij[i][j] = 2.0 * alpha.get(mi, 0) * tvisc.get(mi, 0) * avgSij;
        }
      }

      DblType Pij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          Pij[i][j] = 0.0;
          for (int m = 0; m < nDim; ++m) {
            Pij[i][j] += avgDudx.get(mi, i * nDim + m) * tij[j][m] +
                         avgDudx.get(mi, j * nDim + m) * tij[i][m];
          }
          Pij[i][j] *= 0.5;
        }
      }

      DblType P_res = 0.0;
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          P_res += avgDudx.get(mi, i * nDim + j) *
                   (density.get(mi, 0) * (avgVel.get(mi, i) - vel.get(mi, i)) *
                    (avgVel.get(mi, j) - vel.get(mi, j)));
        }
      }

      DblType instProd = 0.0;
      for (int i = 0; i < nDim; ++i)
        instProd += Pij[i][i];

      instProd -= P_res;

      // TODO: Allow for a different averaging timescale for production
      avgProd.get(mi, 0) =
        weightAvg * avgProd.get(mi, 0) + weightInst * instProd;

      // get Mij field_data
      DblType p_Mij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType PM[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType Q[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType D[nalu_ngp::NDimMax][nalu_ngp::NDimMax];

      for (int i = 0; i < nDim; i++) {
        const int iNdim = i * nDim;
        for (int j = 0; j < nDim; j++) {
          p_Mij[i][j] = Mij.get(mi, iNdim + j);
        }
      }

      // Eigenvalue decomposition of metric tensor
      EigenDecomposition::sym_diagonalize<DblType>(p_Mij, Q, D);

      // initialize M43 to 0
      DblType M43[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          M43[i][j] = 0.0;

      const DblType fourThirds = 4.0 / 3.0;

      for (int l = 0; l < nDim; l++) {
        const DblType D43 = stk::math::pow(D[l][l], fourThirds);
        for (int i = 0; i < nDim; i++) {
          for (int j = 0; j < nDim; j++) {
            M43[i][j] += Q[i][l] * Q[j][l] * D43;
          }
        }
      }

      // zeroing out tensors
      DblType tauSGRS[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType tauSGET[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType tau[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      DblType Psgs[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          tauSGRS[i][j] = 0.0;
          tauSGET[i][j] = 0.0;
          tau[i][j] = 0.0;
          Psgs[i][j] = 0.0;
        }
      }

      const DblType CM43 =
        tams_utils::get_M43_constant<DblType, nalu_ngp::NDimMax>(D, CMdeg);

      const DblType epsilon13 =
        stk::math::pow(betaStar * tke.get(mi, 0) * sdr.get(mi, 0), 1.0 / 3.0);

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          // Calculate tauSGRS_ij = 2*alpha*nu_t*<S_ij> where nu_t comes from
          // the SST model and <S_ij> is the strain rate tensor based on the
          // mean quantities... i.e this is (tauSGRS = alpha*tauSST)
          // The 2 in the coeff cancels with the 1/2 in the strain rate tensor
          const DblType coeffSGRS = alpha.get(mi, 0) * tvisc.get(mi, 0);
          tauSGRS[i][j] =
            avgDudx.get(mi, i * nDim + j) + avgDudx.get(mi, j * nDim + i);
          tauSGRS[i][j] *= coeffSGRS;

          for (int l = 0; l < nDim; ++l) {
            // Calculate tauSGET_ij = CM43*<eps>^(1/3)*(M43_ik*dkuj' +
            // M43_jkdkui') where <eps> is the mean dissipation backed out from
            // the SST mean k and mean omega and dkuj' is the fluctuating
            // velocity gradients.
            const DblType coeffSGET = density.get(mi, 0) * CM43 * epsilon13;
            const DblType fluctDudx_jl =
              dudx.get(mi, j * nDim + l) - avgDudx.get(mi, j * nDim + l);
            const DblType fluctDudx_il =
              dudx.get(mi, i * nDim + l) - avgDudx.get(mi, i * nDim + l);
            tauSGET[i][j] +=
              coeffSGET * (M43[i][l] * fluctDudx_jl + M43[j][l] * fluctDudx_il);
          }
        }
      }

      // Calculate the full subgrid stress including the isotropic portion
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          tau[i][j] = tauSGRS[i][j] + tauSGET[i][j] -
                      ((i == j) ? 2.0 / 3.0 * density.get(mi, 0) *
                                    alpha.get(mi, 0) * tke.get(mi, 0)
                                : 0.0);

      // Calculate the SGS production PSGS_ij = 1/2(tau_ik*djuk + tau_jk*diuk)
      // where diuj is the instantaneous velocity gradients
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          for (int l = 0; l < nDim; ++l) {
            Psgs[i][j] += tau[i][l] * dudx.get(mi, l * nDim + j) +
                          tau[j][l] * dudx.get(mi, l * nDim + i);
          }
          Psgs[i][j] *= 0.5;
        }
      }

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          PM[i][j] = 0.0;
          for (int l = 0; l < nDim; ++l)
            PM[i][j] += Psgs[i][l] * p_Mij[l][j];
        }
      }

      // Scale PM first
      const DblType v2 =
        1.0 / v2cMu *
        (tvisc.get(mi, 0) / density.get(mi, 0) / avgTime.get(mi, 0));
      const DblType PMscale = stk::math::pow(1.5 * alpha.get(mi, 0) * v2, -1.5);

      // Handle case where tke = 0, should only occur at a wall boundary
      if (tke.get(mi, 0) == 0.0)
        resAdeq.get(mi, 0) = 1.0;
      else {
        for (int i = 0; i < nDim; ++i)
          for (int j = 0; j < nDim; ++j)
            PM[i][j] = PM[i][j] * PMscale;

        // FIXME: PM is not symmetric
        EigenDecomposition::unsym_matrix_force_sym<DblType>(PM, Q, D);

        const DblType maxPM = stk::math::max(
          stk::math::abs(D[0][0]), stk::math::max(stk::math::abs(D[1][1]), stk::math::abs(D[2][2])));

        // Update the instantaneous resAdeq field
        resAdeq.get(mi, 0) = maxPM;

        resAdeq.get(mi, 0) = stk::math::min(resAdeq.get(mi, 0), 30.0);

        if (alpha.get(mi, 0) >= 1.0)
          resAdeq.get(mi, 0) = stk::math::min(resAdeq.get(mi, 0), 1.0);
      }
      avgResAdeq.get(mi, 0) =
        weightAvg * avgResAdeq.get(mi, 0) + weightInst * resAdeq.get(mi, 0);
    });
}

} // namespace nalu
} // namespace sierra
