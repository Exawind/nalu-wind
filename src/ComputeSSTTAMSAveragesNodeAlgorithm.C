/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeSSTTAMSAveragesNodeAlgorithm.h>
#include <EigenDecomposition.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

#include "utils/TAMSUtils.h"

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeSSTTAMSAveragesNodeAlgorithm - TAMS average quantities for SST 
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeSSTTAMSAveragesNodeAlgorithm::ComputeSSTTAMSAveragesNodeAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    CMdeg_(realm.get_turb_model_constant(TM_CMdeg)),
    meshMotion_(realm_.does_mesh_move())
{
  // save off data
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // instantaneous quantities
  if (meshMotion_) {
   throw std::runtime_error("SSTTAMSAverages: TAMS is not set up to handle mesh motion yet");  
   //velocityRTM_ = meta_data.get_field<VectorFieldType>(
   //  stk::topology::NODE_RANK, "velocity_rtm");
  } 
  else {
    velocityRTM_ = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity");
  }
  density_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  dudx_ = meta_data.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "dudx");
  resAdeq_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter");

  // average quantities
  // FIXME: Need to configure averages to work with mesh motion
  turbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_ke");
  specDissipationRate_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "specific_dissipation_rate");
  avgVelocity_ = meta_data.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "average_velocity");
  avgDensity_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_density");
  avgDudx_ = meta_data.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "average_dudx");
  avgTkeRes_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_tke_resolved");
  avgProd_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_production");
  avgTime_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_time");
  avgResAdeq_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "avg_res_adequacy_parameter");

  // Other quantities
  tvisc_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  alpha_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "k_ratio");
  Mij_ = meta_data.get_field<GenericFieldType>(
    stk::topology::NODE_RANK, "metric_tensor");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeSSTTAMSAveragesNodeAlgorithm::execute()
{
  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // time step
  const double dt = realm_.get_time_step();

  tauSGET.resize(nDim * nDim);
  tauSGRS.resize(nDim * nDim);
  tau.resize(nDim * nDim);
  Psgs.resize(nDim * nDim);

  // pointers to the local storage vectors
  double* p_tauSGET = &tauSGET[0];
  double* p_tauSGRS = &tauSGRS[0];
  double* p_tau = &tau[0];
  double* p_Psgs = &Psgs[0];

  // deal with state 
  // FIXME: Do i need this? is StateNP1 the default? 
  VectorFieldType& velocityNp1 = velocityRTM_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType& densityNp1 = density_->field_of_state(stk::mesh::StateNP1);
  //ScalarFieldType& tkeNp1 = turbKineticEnergy_->field_of_state(stk::mesh::StateNP1);

  // fill in nodal values
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*avgVelocity_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // get instantaneous field data
    const double* rho = stk::mesh::field_data(densityNp1, b);

    // get other field data
    const double* tvisc = stk::mesh::field_data(*tvisc_, b);
    double* alpha = stk::mesh::field_data(*alpha_, b);

    // get average field data
    const double* tke = stk::mesh::field_data(*turbKineticEnergy_, b);
    const double* sdr = stk::mesh::field_data(*specDissipationRate_, b);
    double* avgRho = stk::mesh::field_data(*avgDensity_, b);
    double* avgProd = stk::mesh::field_data(*avgProd_, b);
    double* avgTkeRes = stk::mesh::field_data(*avgTkeRes_, b);
    double* avgTime = stk::mesh::field_data(*avgTime_, b);

    // Get resolution adequacy field for filling
    double* resAdeq = stk::mesh::field_data(*resAdeq_, b);
    double* avgResAdeq = stk::mesh::field_data(*avgResAdeq_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get velocity field data
      const double* vel = stk::mesh::field_data(velocityNp1, b[k]);
      const double* dudx = stk::mesh::field_data(*dudx_, b[k]);
      double* avgVel = stk::mesh::field_data(*avgVelocity_, b[k]);
      double* avgDudx = stk::mesh::field_data(*avgDudx_, b[k]);

      // store RANS time scale
      avgTime[k] = 1.0 / (betaStar_ * sdr[k]);

      // causal time average ODE: d<phi>/dt = 1/avgTime * (phi - <phi>)
      const double weightAvg = std::max(1.0 - dt / avgTime[k], 0.0);
      const double weightInst = std::min(dt / avgTime[k], 1.0);

      for (int i = 0; i < nDim; ++i)
        avgVel[i] = weightAvg * avgVel[i] + weightInst * vel[i];

      double tkeRes = 0.0;
      for (int i = 0; i < nDim; ++i) {
        tkeRes += (vel[i] - avgVel[i]) * (vel[i] - avgVel[i]);
        for (int j = 0; j < nDim; ++j) {
          avgDudx[i * nDim + j] =
            weightAvg * avgDudx[i * nDim + j] + weightInst * dudx[i * nDim + j];
        }
      }

      // TODO: Do I need density weighted averaging when density varies?
      avgRho[k] = weightAvg * avgRho[k] + weightInst * rho[k];
      avgTkeRes[k] = weightAvg * avgTkeRes[k] + weightInst * 0.5 * tkeRes;

      // Calculate alpha
      if (tke[k] == 0.0)
        alpha[k] = 1.0;
      else {
        alpha[k] = 1.0 - avgTkeRes[k] / tke[k];

        // limiters
        alpha[k] = std::min(alpha[k], 1.0);

        // FIXME: What to do with a_kol in SST?
        const double a_kol = 0.01;

        alpha[k] = std::max(alpha[k], a_kol);
      }

      // Production averaging
      double tij[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          const double avgSij = 0.5 * (avgDudx[i * nDim + j] + 
                                       avgDudx[j * nDim + i]);
          tij[i][j] = 2.0 * alpha[k] * tvisc[k] * avgSij;
        }
        tij[i][i] -= 2.0 / 3.0 * alpha[k] * tke[k] * avgRho[k];
      }

      double Pij[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          Pij[i][j] = 0.0;
          for (int m = 0; m < nDim; ++m) {
            Pij[i][j] += avgDudx[i * nDim + m] * tij[j][m] +
                         avgDudx[j * nDim + m] * tij[i][m];
          }
          Pij[i][j] *= 0.5;
        }
      }

      double P_res = 0.0;
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          P_res += avgDudx[i * nDim + j] *
                   (avgRho[k] * (avgVel[i] - vel[i]) * (avgVel[j] - vel[j]));
        }
      }

      double instProd = 0.0;
      for (int i = 0; i < nDim; ++i)
        instProd += Pij[i][i];

      instProd -= P_res;

      // TODO: Allow for a different averaging timescale for production
      avgProd[k] = weightAvg * avgProd[k] + weightInst * instProd;

      // get Mij field_data
      const double* p_Mij = stk::mesh::field_data(*Mij_, b[k]);

      double Mij[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      double PM[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      double Q[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      double D[nalu_ngp::nDimMax][nalu_ngp::nDimMax];

      for (int i = 0; i < nDim; i++) {
        const int iNdim = i * nDim;
        for (int j = 0; j < nDim; j++) {
          Mij[i][j] = p_Mij[iNdim + j];
        }
      }

      // Eigenvalue decomposition of metric tensor
      EigenDecomposition::sym_diagonalize<double>(Mij, Q, D);

      // initialize M43 to 0
      double M43[nalu_ngp::nDimMax][nalu_ngp::nDimMax];
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          M43[i][j] = 0.0;

      const double fourThirds = 4.0 / 3.0;

      for (int l = 0; l < nDim; l++) {
        const double D43 = stk::math::pow(D[l][l], fourThirds);
        for (int i = 0; i < nDim; i++) {
          for (int j = 0; j < nDim; j++) {
            M43[i][j] += Q[i][l] * Q[j][l] * D43;
          }
        }
      }

      // zeroing out tesnors
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          p_tauSGRS[i * nDim + j] = 0.0;
          p_tauSGET[i * nDim + j] = 0.0;
          p_tau[i * nDim + j] = 0.0;
          p_Psgs[i * nDim + j] = 0.0;
        }
      }

      const double CM43 = tams_utils::get_M43_constant<double, nalu_ngp::nDimMax>(D, CMdeg_);

      const double epsilon13 =
        stk::math::pow(betaStar_ * tke[k] * sdr[k], 1.0 / 3.0);

      for (int i = 0; i < nDim; ++i) {

        for (int j = 0; j < nDim; ++j) {
          // Calculate tauSGRS_ij = 2*alpha*nu_t*<S_ij> where nu_t comes from
          // the SST model and <S_ij> is the strain rate tensor based on the
          // mean quantities... i.e this is (tauSGRS = alpha*tauSST)
          // The 2 in the coeff cancels with the 1/2 in the strain rate tensor
          const double coeffSGRS = alpha[k] * tvisc[k];
          p_tauSGRS[i * nDim + j] = avgDudx[i * nDim + j] + avgDudx[j * nDim + i];
          p_tauSGRS[i * nDim + j] *= coeffSGRS;

          for (int l = 0; l < nDim; ++l) {
            // Calculate tauSGET_ij = CM43*<eps>^(1/3)*(M43_ik*dkuj' +
            // M43_jkdkui') where <eps> is the mean dissipation backed out from
            // the SST mean k and mean omega and dkuj' is the fluctuating
            // velocity gradients.
            const double coeffSGET = avgRho[k] * CM43 * epsilon13;
            const double fluctDudx_jl = dudx[j * nDim + l] - avgDudx[j * nDim + l];
            const double fluctDudx_il = dudx[i * nDim + l] - avgDudx[i * nDim + l];
            p_tauSGET[i * nDim + j] +=
              coeffSGET * (M43[i][l] * fluctDudx_jl + M43[j][l] * fluctDudx_il);
          }
        }
      }

      // Calculate the full subgrid stress including the isotropic portion
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          p_tau[i * nDim + j] = p_tauSGRS[i * nDim + j] + p_tauSGET[i * nDim + j] -
            ((i == j) ? 2.0 / 3.0 * avgRho[k] * alpha[k] * tke[k] : 0.0);

      // Calculate the SGS production PSGS_ij = 1/2(tau_ik*djuk + tau_jk*diuk)
      // where diuj is the instantaneous velocity gradients
      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          for (int l = 0; l < nDim; ++l) {
            p_Psgs[i * nDim + j] += p_tau[i * nDim + l] * dudx[l * nDim + j] +
                                     p_tau[j * nDim + l] * dudx[l * nDim + i];
          }
          p_Psgs[i * nDim + j] *= 0.5;
        }
      }

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          PM[i][j] = 0.0;
          for (int l = 0; l < nDim; ++l)
            PM[i][j] += p_Psgs[i * nDim + l] * Mij[l][j];
        }
      }

      // Scale PM first
      const double v2 = 1.0 / 0.22 * (tvisc[k] / avgRho[k] / avgTime[k]);
      const double PMscale = std::pow(1.5 * alpha[k] * v2, -1.5);

      // Handle case where tke = 0, should only occur at a wall boundary
      if (tke[k] == 0.0)
        resAdeq[k] = 1.0;
      else {
        for (int i = 0; i < nDim; ++i)
          for (int j = 0; j < nDim; ++j)
            PM[i][j] = PM[i][j] * PMscale;

        // FIXME: PM is not symmetric
        EigenDecomposition::unsym_matrix_force_sym<double>(PM, Q, D);

        const double maxPM = std::max(std::abs(D[0][0]),
                                      std::max(std::abs(D[1][1]), std::abs(D[2][2])));

        // Update the instantaneous resAdeq field
        resAdeq[k] = maxPM;

        resAdeq[k] = std::min(resAdeq[k], 30.0);

        if (alpha[k] >= 1.0)
          resAdeq[k] = std::min(resAdeq[k], 1.0);
      }
      avgResAdeq[k] = weightAvg * avgResAdeq[k] + weightInst * resAdeq[k];
    }
  }
}

} // namespace nalu
} // namespace sierra
