// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/MomentumSSTLRAMSDiffEdgeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "EigenDecomposition.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include "utils/AMSUtils.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

MomentumSSTLRAMSDiffEdgeKernel::MomentumSSTLRAMSDiffEdgeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPEdgeKernel<MomentumSSTLRAMSDiffEdgeKernel>(),
    includeDivU_(solnOpts.includeDivU_),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    aspectRatioSwitch_(solnOpts.get_turb_model_constant(TM_aspRatSwitch)),
    alphaPow_(solnOpts.get_turb_model_constant(TM_alphaPow)),
    alphaScaPow_(solnOpts.get_turb_model_constant(TM_alphaScaPow)),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  edgeAreaVecID_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);

  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  velocityID_ = get_field_ordinal(meta, "velocity");
  turbViscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  viscID_ = get_field_ordinal(meta, "viscosity");
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1ID_ =
    get_field_ordinal(meta, "specific_dissipation_rate", stk::mesh::StateNP1);
  betaID_ = get_field_ordinal(meta, "k_ratio");
  MijID_ = get_field_ordinal(meta, "metric_tensor");
  fOneBlendID_ = get_field_ordinal(meta, "sst_f_one_blending");
  dudxID_ = get_field_ordinal(meta, "dudx");

  // average quantities
  avgVelocityID_ = get_field_ordinal(meta, "average_velocity");
  avgDudxID_ = get_field_ordinal(meta, "average_dudx");
  avgResAdeqID_ = get_field_ordinal(meta, "avg_res_adequacy_parameter");

  const std::string dofName = "velocity";
  relaxFacU_ = solnOpts.get_relaxation_factor(dofName);
}

void
MomentumSSTLRAMSDiffEdgeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  edgeAreaVec_ = fieldMgr.get_field<double>(edgeAreaVecID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  velocity_ = fieldMgr.get_field<double>(velocityID_);
  tvisc_ = fieldMgr.get_field<double>(turbViscID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  density_ = fieldMgr.get_field<double>(densityNp1ID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  beta_ = fieldMgr.get_field<double>(betaID_);
  nodalMij_ = fieldMgr.get_field<double>(MijID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  avgVelocity_ = fieldMgr.get_field<double>(avgVelocityID_);
  avgDudx_ = fieldMgr.get_field<double>(avgDudxID_);
  avgResAdeq_ = fieldMgr.get_field<double>(avgResAdeqID_);
}

void
MomentumSSTLRAMSDiffEdgeKernel::execute(
  EdgeKernelTraits::ShmemDataType& smdata,
  const stk::mesh::FastMeshIndex& edge,
  const stk::mesh::FastMeshIndex& nodeL,
  const stk::mesh::FastMeshIndex& nodeR)
{
  const int ndim = nDim_;

  // Scratch work arrays
  NALU_ALIGNED EdgeKernelTraits::DblType av[EdgeKernelTraits::NDimMax];

  for (int d = 0; d < nDim_; d++) {
    av[d] = edgeAreaVec_.get(edge, d);
  }

  // Mij, eigenvectors and eigenvalues
  EdgeKernelTraits::DblType Mij[EdgeKernelTraits::NDimMax]
                               [EdgeKernelTraits::NDimMax];
  EdgeKernelTraits::DblType Q[EdgeKernelTraits::NDimMax]
                             [EdgeKernelTraits::NDimMax];
  EdgeKernelTraits::DblType D[EdgeKernelTraits::NDimMax]
                             [EdgeKernelTraits::NDimMax];

  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      Mij[i][j] = 0.5 * (nodalMij_.get(nodeL, i * ndim + j) +
                         nodalMij_.get(nodeR, i * ndim + j));

  EigenDecomposition::sym_diagonalize<EdgeKernelTraits::DblType>(Mij, Q, D);

  // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
  // so to create M43, we use Q D^(4/3) Q^T
  EdgeKernelTraits::DblType M43[EdgeKernelTraits::NDimMax]
                               [EdgeKernelTraits::NDimMax];
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      M43[i][j] = 0.0;

  const double fourThirds = 4. / 3.;
  for (int k = 0; k < ndim; k++) {
    const EdgeKernelTraits::DblType D43 = stk::math::pow(D[k][k], fourThirds);
    for (int i = 0; i < ndim; i++) {
      for (int j = 0; j < ndim; j++) {
        M43[i][j] += Q[i][k] * Q[j][k] * D43;
      }
    }
  }

  // Compute cell aspect ratio blending
  const EdgeKernelTraits::DblType maxEigM =
    stk::math::max(D[0][0], stk::math::max(D[1][1], D[2][2]));
  const EdgeKernelTraits::DblType minEigM =
    stk::math::min(D[0][0], stk::math::min(D[1][1], D[2][2]));
  const EdgeKernelTraits::DblType aspectRatio = maxEigM / minEigM;

  const EdgeKernelTraits::DblType arScale =
    1.0 - stk::math::tanh(stk::math::max(
            0.5 * (avgResAdeq_.get(nodeL, 0) + avgResAdeq_.get(nodeR, 0)) - 1.0,
            0.0));

  const EdgeKernelTraits::DblType arInvScale = 1.0 - arScale;

  // Compute CM43
  EdgeKernelTraits::DblType CM43 = ams_utils::get_M43_constant<
    EdgeKernelTraits::DblType, EdgeKernelTraits::NDimMax>(D, CMdeg_);

  const EdgeKernelTraits::DblType CM43scale = stk::math::max(
    stk::math::min(
      0.5 * (stk::math::pow(avgResAdeq_.get(nodeL, 0), 2.0) +
             stk::math::pow(avgResAdeq_.get(nodeR, 0), 2.0)),
      30.0),
    1.0);

  const EdgeKernelTraits::DblType muIp =
    0.5 * (tvisc_.get(nodeL, 0) + tvisc_.get(nodeR, 0));
  const EdgeKernelTraits::DblType rhoIp =
    0.5 * (density_.get(nodeL, 0) + density_.get(nodeR, 0));
  const EdgeKernelTraits::DblType tkeIp =
    0.5 * (stk::math::max(tke_.get(nodeL, 0), 1.0e-12) +
           stk::math::max(tke_.get(nodeR, 0), 1.0e-12));
  const EdgeKernelTraits::DblType sdrIp =
    0.5 * (stk::math::max(sdr_.get(nodeL, 0), 1.0e-12) +
           stk::math::max(sdr_.get(nodeR, 0), 1.0e-12));
  const EdgeKernelTraits::DblType alphaIp =
    0.5 * (stk::math::pow(beta_.get(nodeL, 0), alphaPow_) +
           stk::math::pow(beta_.get(nodeR, 0), alphaPow_));
  const EdgeKernelTraits::DblType fOneBlendIp =
    0.5 * (fOneBlend_.get(nodeL, 0) + fOneBlend_.get(nodeR, 0));
  const EdgeKernelTraits::DblType molViscIp =
    0.5 * (visc_.get(nodeL, 0) + visc_.get(nodeR, 0));

  EdgeKernelTraits::DblType avgdUidxj[EdgeKernelTraits::NDimMax]
                                     [EdgeKernelTraits::NDimMax];
  EdgeKernelTraits::DblType fluctdUidxj[EdgeKernelTraits::NDimMax]
                                       [EdgeKernelTraits::NDimMax];

  EdgeKernelTraits::DblType axdx = 0.0;
  EdgeKernelTraits::DblType asq = 0.0;
  for (int d = 0; d < ndim; ++d) {
    const EdgeKernelTraits::DblType dxj =
      coordinates_.get(nodeR, d) - coordinates_.get(nodeL, d);
    asq += av[d] * av[d];
    axdx += av[d] * dxj;
  }
  const EdgeKernelTraits::DblType inv_axdx = 1.0 / axdx;

  // Compute average divU
  for (int i = 0; i < ndim; ++i) {

    // difference between R and L nodes for component i
    const EdgeKernelTraits::DblType avgUidiff =
      avgVelocity_.get(nodeR, i) - avgVelocity_.get(nodeL, i);
    const EdgeKernelTraits::DblType fluctUidiff =
      (velocity_.get(nodeR, i) - velocity_.get(nodeL, i)) - avgUidiff;

    const int offSetI = ndim * i;

    // start sum for NOC contribution
    EdgeKernelTraits::DblType GlavgUidxl = 0.0;
    EdgeKernelTraits::DblType GlfluctUidxl = 0.0;
    for (int l = 0; l < ndim; ++l) {
      const int offSetIL = offSetI + l;
      const EdgeKernelTraits::DblType dxl =
        coordinates_.get(nodeR, l) - coordinates_.get(nodeL, l);
      const EdgeKernelTraits::DblType GlavgUi =
        0.5 * (avgDudx_.get(nodeL, offSetIL) + avgDudx_.get(nodeR, offSetIL));
      const EdgeKernelTraits::DblType GlfluctUi =
        0.5 * (dudx_.get(nodeL, offSetIL) + dudx_.get(nodeR, offSetIL)) -
        GlavgUi;
      GlavgUidxl += GlavgUi * dxl;
      GlfluctUidxl += GlfluctUi * dxl;
    }

    // form full tensor dui/dxj with NOC
    for (int j = 0; j < ndim; ++j) {
      const int offSetIJ = offSetI + j;
      const EdgeKernelTraits::DblType GjavgUi =
        0.5 * (avgDudx_.get(nodeL, offSetIJ) + avgDudx_.get(nodeR, offSetIJ));
      const EdgeKernelTraits::DblType GjfluctUi =
        0.5 * (dudx_.get(nodeL, offSetIJ) + dudx_.get(nodeR, offSetIJ)) -
        GjavgUi;
      avgdUidxj[i][j] = GjavgUi + (avgUidiff - GlavgUidxl) * av[j] * inv_axdx;
      fluctdUidxj[i][j] =
        GjfluctUi + (fluctUidiff - GlfluctUidxl) * av[j] * inv_axdx;
    }
  }

  EdgeKernelTraits::DblType avgDivU = 0.0;
  for (int i = 0; i < ndim; ++i) {
    avgDivU += avgdUidxj[i][i];
  }

  const EdgeKernelTraits::DblType ReT = rhoIp * tkeIp / sdrIp / molViscIp;
  const EdgeKernelTraits::DblType Rbeta = 8.0;
  const EdgeKernelTraits::DblType betaStarLowRe =
    betaStar_ * (4.0 / 15.0 + stk::math::pow(ReT / Rbeta, 4.0)) /
    (1.0 + stk::math::pow(ReT / Rbeta, 4.0));
  const EdgeKernelTraits::DblType betaStarBlend =
    fOneBlendIp * betaStarLowRe + (1.0 - fOneBlendIp) * betaStar_;

  const EdgeKernelTraits::DblType epsilon13Ip =
    stk::math::pow(betaStarBlend * tkeIp * sdrIp, 1.0 / 3.0);

  for (int i = 0; i < ndim; ++i) {
    // Left and right row/col indices
    const int rowL = i;
    const int rowR = i + ndim;

    // This is the divU term for the average quantities in the model for
    // tau_ij^SGRS Since we are letting SST calculate it's normal mu_t, we
    // need to scale by alpha here
    const EdgeKernelTraits::DblType avgDivUstress =
      2.0 / 3.0 * stk::math::pow(alphaIp, alphaScaPow_) * (2.0 - alphaIp) *
      muIp * avgDivU * av[i] * includeDivU_;
    smdata.rhs(rowL) -= avgDivUstress;
    smdata.rhs(rowR) += avgDivUstress;

    EdgeKernelTraits::DblType lhs_riC_i = 0.0;
    EdgeKernelTraits::DblType lhs_riC_SGRS_i = 0.0;
    for (int j = 0; j < ndim; ++j) {

      // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
      EdgeKernelTraits::DblType rhsfacDiff_i = 0.0;
      EdgeKernelTraits::DblType lhsfacDiff_i = 0.0;
      for (int k = 0; k < ndim; ++k) {
        lhsfacDiff_i += -rhoIp * CMdeg_ * epsilon13Ip * arScale * M43[j][k] *
                        av[k] * av[j] * inv_axdx;
        rhsfacDiff_i += -rhoIp * CMdeg_ * epsilon13Ip * arScale * M43[j][k] *
                        fluctdUidxj[i][k] * av[j];
      }

      lhsfacDiff_i += -arInvScale * muIp * av[j] * av[j] * inv_axdx;
      rhsfacDiff_i += -arInvScale * muIp * fluctdUidxj[i][j] * av[j];

      // Accumulate lhs
      lhs_riC_i += lhsfacDiff_i;

      // SGRS (average) term, scaled by alpha
      const EdgeKernelTraits::DblType rhsSGRSfacDiff_i =
        -stk::math::pow(alphaIp, alphaScaPow_) * (2.0 - alphaIp) * muIp *
        avgdUidxj[i][j] * av[j];

      // Implicit treatment of SGRS (average) term
      lhs_riC_SGRS_i += -stk::math::pow(alphaIp, alphaScaPow_) *
                        (2.0 - alphaIp) * muIp * av[j] * av[j] * inv_axdx;

      smdata.rhs(rowL) -= rhsfacDiff_i + rhsSGRSfacDiff_i;
      smdata.rhs(rowR) += rhsfacDiff_i + rhsSGRSfacDiff_i;

      // -mut^ik*duj/dxk*A_j
      EdgeKernelTraits::DblType rhsfacDiff_j = 0.0;
      EdgeKernelTraits::DblType lhsfacDiff_j = 0.0;
      for (int k = 0; k < ndim; ++k) {
        lhsfacDiff_j += -rhoIp * CMdeg_ * epsilon13Ip * arScale * M43[i][k] *
                        av[k] * av[j] * inv_axdx;
        rhsfacDiff_j += -rhoIp * CMdeg_ * epsilon13Ip * arScale * M43[i][k] *
                        fluctdUidxj[j][k] * av[j];
      }

      lhsfacDiff_j += -arInvScale * muIp * av[i] * av[j] * inv_axdx;
      rhsfacDiff_j += -arInvScale * muIp * fluctdUidxj[j][i] * av[j];

      // SGRS (average) term, scaled by alpha
      const EdgeKernelTraits::DblType rhsSGRSfacDiff_j =
        -stk::math::pow(alphaIp, alphaScaPow_) * (2.0 - alphaIp) * muIp *
        avgdUidxj[j][i] * av[j];

      // Implicit treatment of SGRS (average) term
      const EdgeKernelTraits::DblType lhsSGRSfacDiff_j =
        -stk::math::pow(alphaIp, alphaScaPow_) * (2.0 - alphaIp) * muIp *
        av[i] * av[j] * inv_axdx;

      smdata.rhs(rowL) -= rhsfacDiff_j + rhsSGRSfacDiff_j;
      smdata.rhs(rowR) += rhsfacDiff_j + rhsSGRSfacDiff_j;

      const int colL = j;
      const int colR = j + ndim;

      smdata.lhs(rowL, colL) -= (lhsfacDiff_j + lhsSGRSfacDiff_j) / relaxFacU_;
      smdata.lhs(rowL, colR) += (lhsfacDiff_j + lhsSGRSfacDiff_j);
      smdata.lhs(rowR, colL) += (lhsfacDiff_j + lhsSGRSfacDiff_j);
      smdata.lhs(rowR, colR) -= (lhsfacDiff_j + lhsSGRSfacDiff_j) / relaxFacU_;
    }

    smdata.lhs(rowL, rowL) -= (lhs_riC_i + lhs_riC_SGRS_i) / relaxFacU_;
    smdata.lhs(rowL, rowR) += (lhs_riC_i + lhs_riC_SGRS_i);
    smdata.lhs(rowR, rowL) += (lhs_riC_i + lhs_riC_SGRS_i);
    smdata.lhs(rowR, rowR) -= (lhs_riC_i + lhs_riC_SGRS_i) / relaxFacU_;
  }
}

} // namespace nalu
} // namespace sierra
