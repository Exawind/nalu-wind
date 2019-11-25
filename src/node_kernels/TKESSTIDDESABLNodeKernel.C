/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TKESSTIDDESABLNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "SimdInterface.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

TKESSTIDDESABLNodeKernel::TKESSTIDDESABLNodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<TKESSTIDDESABLNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    maxLenScaleID_(get_field_ordinal(meta, "sst_max_length_scale")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    nDim_(meta.spatial_dimension())
{}

void
TKESSTIDDESABLNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  wallDist_        = fieldMgr.get_field<double>(wallDistID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  maxLenScale_     = fieldMgr.get_field<double>(maxLenScaleID_);
  fOneBlend_       = fieldMgr.get_field<double>(fOneBlendID_);

  const std::string dofName = "turbulent_ke";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  cDESke_ = realm.get_turb_model_constant(TM_cDESke);
  cDESkw_ = realm.get_turb_model_constant(TM_cDESkw);
  kappa_ = realm.get_turb_model_constant(TM_kappa);
  iddes_Cw_ = realm.get_turb_model_constant(TM_iddes_Cw);
  iddes_Cdt1_ = realm.get_turb_model_constant(TM_iddes_Cdt1);
  iddes_Cdt2_ = realm.get_turb_model_constant(TM_iddes_Cdt2);
  iddes_Cl_ = realm.get_turb_model_constant(TM_iddes_Cl);
  iddes_Ct_ = realm.get_turb_model_constant(TM_iddes_Ct);
  cEps_ = realm.get_turb_model_constant(TM_cEps);
  abl_bndtw_ = realm.get_turb_model_constant(TM_abl_bndtw);
  abl_deltandtw_ = realm.get_turb_model_constant(TM_abl_deltandtw);
}

void TKESSTIDDESABLNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType dw = wallDist_.get(node,0);
  const DblType maxLenScale = maxLenScale_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

  DblType Pk = 0.0;
  DblType sijSq = 1.0e-16;
  DblType omegaSq = 1.0e-16;
  for (int i=0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j=0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset+j);
      Pk += dudxij * (dudxij + dudx_.get(node, j*nDim_ + i));
      const DblType rateOfStrain =
          0.5*(dudxij
               + dudx_.get(node, j*nDim_ + i));
      sijSq += rateOfStrain*rateOfStrain;
      const DblType rateOfOmega =
          0.5*(dudxij
               - dudx_.get(node, j*nDim_ + i));
      omegaSq += rateOfOmega * rateOfOmega;
    }
  }
  sijSq *= 2.0;
  omegaSq *= 2.0;
  Pk *= tvisc;

  DblType rdl = visc/(density * kappa_ * kappa_ * dw * dw * stk::math::sqrt(0.5 * (sijSq + omegaSq) ));
  DblType rdt = tvisc/(density * kappa_ * kappa_ * dw * dw * (stk::math::sqrt(0.5 * (sijSq + omegaSq) ) + 1e-10) );
  DblType fl = stk::math::tanh( stk::math::pow( iddes_Cl_ * iddes_Cl_ * rdl, 10));
  DblType ft = stk::math::tanh( stk::math::pow( iddes_Cl_ * iddes_Cl_ * rdt, 3));
  DblType alpha = 0.25 - dw/maxLenScale;
  DblType fe1 = (alpha < 0) ? 2.0 * stk::math::exp(-9.0 * alpha * alpha) : 2.0 * stk::math::exp(-11.09 * alpha * alpha);
  DblType fe2 = 1.0 - stk::math::max(ft,fl);
  DblType fe = fe2 * stk::math::max( (fe1 - 1.0), 0.0);
  DblType fb = stk::math::min(2.0 * stk::math::exp(-2.0 * alpha * alpha), 1.0);
  DblType fdt = 1.0 - stk::math::tanh( stk::math::pow(iddes_Cdt1_ * rdt, iddes_Cdt2_) );
  DblType fdHat = stk::math::max( (1.0 - fdt), fb);
  DblType delta = stk::math::min( iddes_Cw_ * stk::math::max(dw, maxLenScale), maxLenScale);
      
  // blend cDES constant
  const DblType cDES =
    fOneBlend * cDESkw_ + (1.0 - fOneBlend) * cDESke_;

  const DblType sqrtTke = stk::math::sqrt(tke);
  const DblType lSST = sqrtTke / betaStar_ / sdr;

  // Find minimum length scale, limit minimum value to 1.0e-16 to prevent
  // division by zero later on
  const DblType lIDDES =
      stk::math::max(1.0e-16, fdHat * (1.0  + fe) * lSST + (1.0 - fdHat) * cDES * delta);

  const DblType f_des_abl = 0.5*stk::math::tanh( (abl_bndtw_ - dw)/abl_deltandtw_) + 0.5;

  const DblType filter = stk::math::cbrt(dVol);
  const DblType lIDDESABL = f_des_abl * lIDDES + (1.0 - f_des_abl) * filter/cEps_;
  
  DblType Dk = density * tke * sqrtTke / lIDDESABL;

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  rhs(0) += (Pk - Dk) * dVol;
  lhs(0, 0) += 1.5 * density / lIDDESABL * sqrtTke * dVol;
}

}  // nalu
}  // sierra
