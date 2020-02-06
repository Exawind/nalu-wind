/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TKESSTBLTNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

TKESSTBLTNodeKernel::TKESSTBLTNodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<TKESSTBLTNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{}

void
TKESSTBLTNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  const std::string dofName = "turbulent_ke";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

void TKESSTBLTNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // Modify Production and destruction TKE source terms with gamma for transition
  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  // const DblType gamint    = gamint_.get(node, 0);
  // const DblType re0t      = re0t_.get(node, 0);
  // const DblType visc      = visc_.get(node, 0);
  // const DblType sijMag    = sijMag_.get(node, 0);
  // const DblType vortMag   = vortMag_.get(node, 0);
  // const DblType minD      = minD_.get(node, 0);

  // NALU_ALIGNED vel[NodeKernelTraits::NDimMax];

  // for (int i=0; i < NodeKernelTraits::NDimMax; ++i) {
  //   vel[i] = velocityNp1_.get(node, i);
  // }

  // DblType Rev = density * minD * minD * sijMag / visc;
  // DblType Re0c = Re_0c(re0t);
  // DblType rt = density * tke / sdr / visc;
  // DblType Freattach = stk::math::exp(-6.25e-6*rt*rt*rt*rt);
  // DblType Reomega = density * sdr * minD * minD / visc;
  // DblType Fwake = stk::math::exp(-1.0e-10 * Reomega * Reomega);
  // DblType velMag = stk::math::sqrt( vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2] ) + 1.e-14;
  // DblType delta = 350.0 * vortMag * minD * re0t * visc / (density * velMag * velMag);
  // DblType farg = minD / delta;
  // DblType crat = (gamint - 1.0/ceTwo)/(1.0-1.0/ceTwo);
  // DblType f0t = stk::math::min(stk::math::max(Fwake*exp(-farg*farg*farg*farg), 1.0 - crat*crat), 1.0);
  // DblType gama_sep = stk::math::min(2.e0*stk::math::max(Rev/3.235e0/Re0c-1.e0,0.e0)*Freattach,2.e0)*f0t;
  // DblType gama_eff = stk::math::max(gama_int, gama_sep);
  DblType gama_eff = 1.0;

  DblType Pk = 0.0;
  for (int i=0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j=0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset+j);
      Pk += dudxij * (dudxij + dudx_.get(node, j*nDim_ + i));
    }
  }
  Pk *= tvisc*gama_eff;

  DblType Dk = betaStar_ * density * sdr * tke * stk::math::min(stk::math::max(gama_eff,0.1),1.0);

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);
  //
  rhs(0) += (Pk - Dk) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * stk::math::min(stk::math::max(gama_eff,0.1),1.0) * dVol;
}

}  // nalu
}  // sierra
