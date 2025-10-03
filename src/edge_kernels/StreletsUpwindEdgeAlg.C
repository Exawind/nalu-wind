// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/StreletsUpwindEdgeAlg.h>
#include <Realm.h>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpFieldUtils.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <SolutionOptions.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

StreletsUpwindEdgeAlg::StreletsUpwindEdgeAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    velocityName_("velocity"),
    pecletFactor_(get_field_ordinal(
      realm_.meta_data(), "peclet_factor", stk::topology::EDGE_RANK)),
    fOne_(get_field_ordinal(realm.meta_data(), "sst_f_one_blending")),
    dualNodalVolume_(get_field_ordinal(realm.meta_data(), "dual_nodal_volume")),
    sstMaxLen_(get_field_ordinal(realm.meta_data(), "sst_max_length_scale")),
    dudx_(get_field_ordinal(realm.meta_data(), "dudx")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    turbViscosity_(get_field_ordinal(realm.meta_data(), "turbulent_viscosity")),
    turbKE_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    specDissRate_(
      get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    edgeAreaVec_(get_field_ordinal(
      realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    coordinates_(get_field_ordinal(
      realm.meta_data(), realm.solutionOptions_->get_coordinates_name())),
    velocity_(get_field_ordinal(realm.meta_data(), velocityName_))
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
                     " alpha should be 0.0 when using IDDES\n";
  if (alphaUpw != 1.0)
    error_message += "alpha_upw is set to: " + std::to_string(alphaUpw) +
                     " alpha_upw should be 1.0 when using IDDES\n";
  if (hoUpwind != 1.0)
    error_message += "upw_factor is set to: " + std::to_string(hoUpwind) +
                     " upw_factor should be 1.0 when using IDDES\n";

  if (!error_message.empty())
    NaluEnv::self().naluOutputP0() << "WARNING::For the momementum equation:\n"
                                   << error_message;
}

void
StreletsUpwindEdgeAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  stk::mesh::MetaData& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto pecFactor = fieldMgr.get_field<double>(pecletFactor_);
  const auto fone = fieldMgr.get_field<double>(fOne_);
  const auto dnv = fieldMgr.get_field<double>(dualNodalVolume_);
  const auto sst_maxlen = fieldMgr.get_field<double>(sstMaxLen_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  const auto rho = fieldMgr.get_field<double>(density_);
  const auto visc = fieldMgr.get_field<double>(viscosity_);
  const auto tvisc = fieldMgr.get_field<double>(turbViscosity_);
  const auto tke = fieldMgr.get_field<double>(turbKE_);
  const auto sdr = fieldMgr.get_field<double>(specDissRate_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vel = fieldMgr.get_field<double>(velocity_);

  const DblType cdes_ke = realm_.get_turb_model_constant(TM_cDESke);
  const DblType cdes_kw = realm_.get_turb_model_constant(TM_cDESkw);
  const DblType cmu = realm_.get_turb_model_constant(TM_cMu);
  const DblType sigmaMax = realm_.get_turb_model_constant(TM_sigmaMax);
  const DblType ch1 = realm_.get_turb_model_constant(TM_ch1);
  const DblType ch2 = realm_.get_turb_model_constant(TM_ch2);
  const DblType ch3 = realm_.get_turb_model_constant(TM_ch3);
  const DblType tau_des = realm_.get_turb_model_constant(TM_tau_des);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    "compute_streletes_des_alpha_upw", ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      const auto& nodes = eInfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);
      const auto edge = eInfo.meshIdx;

      // compute edge quantities
      const DblType rhoEdge = 0.5 * (rho.get(nodeL, 0) + rho.get(nodeR, 0));
      const DblType muEdge = 0.5 * (visc.get(nodeL, 0) + visc.get(nodeR, 0));
      const DblType turbMuEdge =
        0.5 * (tvisc.get(nodeL, 0) + tvisc.get(nodeR, 0));
      const DblType sdrEdge = 0.5 * (sdr.get(nodeL, 0) + sdr.get(nodeR, 0));
      const DblType tkeEdge = 0.5 * (tke.get(nodeL, 0) + tke.get(nodeR, 0));
      const DblType fOneEdge = 0.5 * (fone.get(nodeL, 0) + fone.get(nodeR, 0));
      const DblType sstMaxLenEdge =
        0.5 * (sst_maxlen.get(nodeL, 0) + sst_maxlen.get(nodeR, 0));

      // Scratch work array for edgeAreaVector
      DblType av[nalu_ngp::NDimMax];
      // Populate area vector work array
      for (int d = 0; d < nDim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      // Compute area vector related quantities and (U dot areaVec)
      DblType axdx = 0.0;
      for (int d = 0; d < nDim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      // TODO(psakiev) extract this a funciton into EdgeKernelUtils.h
      // Computation of duidxj term, reproduce original comment by S. P. Domino
      /*
        form duidxj with over-relaxed procedure of Jasak:

        dui/dxj = GjUi +[(uiR - uiL) - GlUi*dxl]*Aj/AxDx
        where Gp is the interpolated pth nodal gradient for ui
      */
      DblType duidxj[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
      for (int i = 0; i < nDim; ++i) {
        const auto dui = vel.get(nodeR, i) - vel.get(nodeL, i);
        const auto offset = i * nDim;

        // Non-orthogonal correction
        DblType gjuidx = 0.0;
        for (int j = 0; j < nDim; ++j) {
          const DblType dxj =
            coordinates.get(nodeR, j) - coordinates.get(nodeL, j);
          const DblType gjui =
            0.5 * (dudx.get(nodeR, offset + j) + dudx.get(nodeL, offset + j));
          gjuidx += gjui * dxj;
        }

        // final dui/dxj with non-orthogonal correction contributions
        for (int j = 0; j < nDim; ++j) {
          const DblType gjui =
            0.5 * (dudx.get(nodeR, offset + j) + dudx.get(nodeL, offset + j));
          duidxj[i][j] = gjui + (dui - gjuidx) * av[j] * inv_axdx;
        }
      }

      DblType sijMag = 0.0;
      DblType omegaMag = 0.0;

      for (int i = 0; i < nDim; i++) {
        for (int j = 0; j < nDim; j++) {
          const DblType rateOfStrain = 0.5 * (duidxj[i][j] + duidxj[j][i]);
          sijMag += rateOfStrain * rateOfStrain;

          const DblType rateOfOmega = 0.5 * (duidxj[i][j] - duidxj[j][i]);
          omegaMag += rateOfOmega * rateOfOmega;
        }
      }

      const DblType ssq_p_osq_o2 = (sijMag + omegaMag);
      sijMag = stk::math::sqrt(2.0 * sijMag);
      omegaMag = stk::math::sqrt(2.0 * omegaMag);

      const DblType B = ch3 * omegaMag * stk::math::max(sijMag, omegaMag) /
                        stk::math::max(ssq_p_osq_o2, 1e-20);
      const DblType g = stk::math::tanh(B * B * B * B);
      const DblType K = stk::math::max(std::sqrt(ssq_p_osq_o2), 0.1 / tau_des);
      const DblType l_turb = stk::math::max(
        (muEdge + turbMuEdge) / rhoEdge /
          stk::math::sqrt(cmu * stk::math::sqrt(cmu) * K),
        stk::math::sqrt(tkeEdge) / (cmu * sdrEdge));
      const DblType cdes = cdes_ke + fOneEdge * (cdes_kw - cdes_ke);
      const DblType A =
        ch2 * stk::math::max(cdes * sstMaxLenEdge / l_turb / g - 0.5, 0.0);

      pecFactor.get(edge, 0) =
        sigmaMax * stk::math::tanh(stk::math::pow(A, ch1));
    });
  pecFactor.modify_on_device();
}

} // namespace nalu
} // namespace sierra
