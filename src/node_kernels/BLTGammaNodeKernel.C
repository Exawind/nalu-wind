/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTGammaNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

BLTGammaNodeKernel::BLTGammaNodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<BLTGammaNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    re0tID_(get_field_ordinal(meta, "Retheta_Eq_transition")), 
    nDim_(meta.spatial_dimension())
{}

void
BLTGammaNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  minD_            = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  re0t_            = fieldMgr.get_field<double>(re0tID_);



  // Update transition model constants
  caOne_ = realm.get_turb_model_constant(TM_caOne);
  caTwo_ = realm.get_turb_model_constant(TM_caTwo);
  ceOne_ = realm.get_turb_model_constant(TM_ceOne);
  ceTwo_ = realm.get_turb_model_constant(TM_ceTwo);
}


double
BLTGammaNodeKernel::Comp_Re_0c(const double& Re0t )
{
    using DblType = NodeKernelTraits::DblType;

    DblType Re0c; // this is the result of this calculation

    if (Re0t <= 1870.0) {   
       Re0c = Re0t - (3.96035e0 - 1.20656e-2*Re0t + 8.6823e-4*Re0t*Re0t - 6.96506e-7*Re0t*Re0t*Re0t + 1.74105e-10*Re0t*Re0t*Re0t*Re0t);
    }
    else {
       Re0c = Re0t - (593.11e0 + 0.482e0*(Re0t - 1870.0e0));
    }

    return Re0c;
}

double
BLTGammaNodeKernel::Comp_f_length(const double& Re0t )
{
    using DblType = NodeKernelTraits::DblType;

    DblType flength; // this is the result of this calculation

    if (Re0t < 400.e0) {
       flength = 39.8189e0 - 1.1927e-2*Re0t - 1.32567e-4*Re0t*Re0t;
    }
    else if (Re0t < 596.e0) {
       flength =263.404 - 1.23939*Re0t + 1.94548e-3*Re0t*Re0t - 1.01695e-6*Re0t*Re0t*Re0t;
    }
    else if (Re0t < 1200.e0) {
       flength = 0.5e0 - 3.e-4*(Re0t - 596.e0);
    }
    else {
      flength = 0.3188e0;
    }

  return flength;
}

void
BLTGammaNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke       = tke_.get(node, 0);
  const DblType sdr       = sdr_.get(node, 0);
  const DblType gamint    = gamint_.get(node, 0);
  const DblType re0t      = re0t_.get(node, 0);
  const DblType density   = density_.get(node, 0);
  const DblType tvisc     = tvisc_.get(node, 0);
  const DblType visc      = visc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol      = dualNodalVolume_.get(node, 0);

  DblType flen = 0.0;
  DblType Re0c = 0.0;
  DblType Romega = 0.0;
  DblType fsublayer = 0.0;
  DblType flength = 0.0;
  DblType Rev = 0.0;
  DblType rt = 0.0;
  
  DblType fonset  = 0.0;
  DblType fonset1 = 0.0;
  DblType fonset2 = 0.0;
  DblType fonset3 = 0.0;
  DblType fturb = 0.0;

  DblType sijMag = 0.0;
  DblType vortMag = 0.0;

  for (int i=0; i < nDim_; ++i) {
    for (int j=0; j < nDim_; ++j) {
          const double duidxj = dudx_.get(node, nDim_ * i + j);
          const double dujdxi = dudx_.get(node, nDim_ * j + i);

          const double rateOfStrain = 0.5 * (duidxj + dujdxi);
          const double vortTensor = 0.5 * (duidxj - dujdxi);
          sijMag += rateOfStrain * rateOfStrain;
          vortMag += vortTensor * vortTensor;
    }
  }

  sijMag = stk::math::sqrt(2.0*sijMag);
  vortMag = stk::math::sqrt(2.0*vortMag);

  flen = Comp_f_length(re0t);
  Re0c = Comp_Re_0c(re0t);
  Romega = density * minD * minD * sdr / 500.0 / visc;
  fsublayer = stk::math::exp(-6.250 * Romega * Romega);
  flength = flen * (1.0 - fsublayer) + 40.0 * fsublayer;
  Rev = density * minD * minD * sijMag / visc;
  fonset1 = Rev/2.1930/Re0c;
  fonset2 = stk::math::min(stk::math::max( fonset1, fonset1*fonset1*fonset1*fonset1),2.0);
  rt = density * tke / sdr / visc;
  fonset3 = stk::math::max(1.0 - 0.0640*rt*rt*rt, 0.0);
  fonset =  stk::math::max(fonset2 - fonset3,0.0);
  fturb =   stk::math::exp(-rt*rt*rt*rt/256.0);
  DblType gamint_arg = stk::math::max( 1.0e-10, fonset * gamint);


  DblType Pgamma = flength * caOne_ * density * sijMag * stk::math::sqrt( stk::math::max( fonset * gamint, 0.0 ) ) * ( 1.0 - ceOne_ * gamint );
  DblType Dgamma = caTwo_ * density * vortMag * gamint * fturb * ( ceTwo_ * gamint - 1.0 );

  rhs(0) += (Pgamma - Dgamma) * dVol;

  DblType t1 = flength * caOne_ * density * sijMag * stk::math::sqrt( gamint * fonset ) * ceOne_;
  DblType t2 = caTwo_ * density * vortMag * fturb * ( 2.0*ceTwo_ * gamint - 1.0 );

  lhs(0, 0) += ( t1 + t2 ) * dVol;
}

} // namespace nalu
}  // sierra
