/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TKESSTBLTM2015NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

TKESSTBLTM2015NodeKernel::TKESSTBLTM2015NodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<TKESSTBLTM2015NodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{}

void
TKESSTBLTM2015NodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  minD_            = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  const std::string dofName = "turbulent_ke";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

void TKESSTBLTM2015NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // Modify Production and destruction TKE source terms with gamma for transition
  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType gamint    = gamint_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc      = visc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  const DblType Ck_BLT = 1.0;
  const DblType CSEP = 1.0;
  const DblType Re0clim = 1100.0;

  DblType sijMag    = 0.0;
  DblType vortMag   = 0.0;
  DblType Pk = 0.0;
  DblType Pklim = 0.0;

  for (int i=0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j=0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset+j);
      Pk += dudxij * (dudxij + dudx_.get(node, j*nDim_ + i));

     const double duidxj = dudx_.get(node, nDim_ * i + j);
     const double dujdxi = dudx_.get(node, nDim_ * j + i);

     const double rateOfStrain = 0.5 * (duidxj + dujdxi);
     const double vortTensor = 0.5 * (duidxj - dujdxi);
     sijMag += rateOfStrain * rateOfStrain;
     vortMag += vortTensor * vortTensor;
    }
  }

  DblType Rev = density * minD*minD * sijMag / visc;
  DblType Fonlim = stk::math::min(stk::math::max(Rev / 2.20 / Re0clim - 1.0, 0.0), 3.0);

// &&&&&&&&&&& hardwire gamint = 1 as first debug step  &&&&&&&&&&&&&&&
   const DblType gamint_debug = 1.0;
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//  Pk = tvisc * sijMag * vortMag; // Pk based on Kato-Launder formulation. Recommended in Menter (2015) to avoid excessive levels of TKE in stagnation regions
  Pk *= tvisc*gamint_debug; // based on standard SST model

  Pklim = 5.0 * Ck_BLT * stk::math::max(gamint_debug - 0.20, 0.0) * (1.0 - gamint_debug) * Fonlim * stk::math::max(3.0 * CSEP * visc - tvisc, 0.0) * sijMag * vortMag;

  if (stk::math::abs(Pklim) > 1.0e-10) {
    std::cout << "Pklim is non-zero: " << Pklim << std::endl;
  }

  DblType Dk = betaStar_ * density * sdr * tke * stk::math::max(gamint_debug,0.1);

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk); // ???????
  //
  rhs(0) += (Pk + Pklim - Dk) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * stk::math::max(gamint_debug,0.1) * dVol;
}

}  // nalu
}  // sierra
