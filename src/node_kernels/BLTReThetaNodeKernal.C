/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTRe0tNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

BLTRe0tNodeKernel::BLTRe0tNodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<BLTRe0tNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "molecular_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    sijMag_(get_field_ordinal(meta, "strainrate_magnitude")),
    vortMag_(get_field_ordinal(meta, "vorticity_magnitude")),
    minD_(get_field_ordinal(meta, "min_wall_distance")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    nDim_(meta.spatial_dimension())
{}

void
BLTRe0tNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  re0t_            = fieldMgr.get_field<double>(re0tID_);

  density_         = fieldMgr.get_field<double>(densityID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  dkdx_            = fieldMgr.get_field<double>(dkdxID_);
  dwdx_            = fieldMgr.get_field<double>(dwdxID_);
  sijMag_          = fieldMgr.get_field<double>(sijMagID_);
  vortMag_         = fieldMgr.get_field<double>(vortMagID_);
  minD_            = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  fOneBlend_       = fieldMgr.get_field<double>(fOneBlendID_);

  const std::string dofName = "specific_dissipation_rate";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
  // Update transition model constants
  caOne_ = realm.get_turb_model_constant(TM_caOne);
  caTwo_ = realm.get_turb_model_constant(TM_caTwo);
  ceOne_ = realm.get_turb_model_constant(TM_ceOne);
  ceTwo_ = realm.get_turb_model_constant(TM_ceTwo);
  c0t_   = realm.get_turb_model_constant(TM_c0t);
}

double
BLTRe0tNodeKernel::Re_thetat( double Tu, double Fla)
{
    using DblType = NodeKernelTraits::DblType;

    const double Tu,Fla;
    double Re0t;

    if (Tu <= 1.3) {   
       Re0t = ( 1173.51 - 589.428 * Tu + 0.2196 / Tu*/Tu) * Fla;
    }
    else {
       Re0t = 331.5 * stk::math::pow(Tu - 0.5658, -0.671) * Fla;
    }

    Re0t = stk::math::max(Re0t, 20.0);

    return Re0t;
}

BLTRe0tNodeKernel::F_lamda( double Tu, double lamda)
{
    using DblType = NodeKernelTraits::DblType;

    const double lamda, Tu;
    double Fla;

    if ( lamda <= 0.0) {  
       Fla = 1.0 + (12.986 * lamda + 123.66 * lamda*lamda + 405.689 * lamda*lamda*lamda) * stk::math::exp(-pow(Tu/1.5, 1.5) );
    }
    else {
       Fla = 1.0 + 0.275 * (1.0-stk::math::exp(-35.0 * lamda)) * stk::math::exp(-2.0*Tu);
    }

    return Fla;
}

double
BLTRe0tNodeKernel::Secant_Re0tcor( double duds, double dens_local, double Tu, double visc)
{
    using DblType = NodeKernelTraits::DblType;

    const int itmax = 10;
    int iloop;

    const double duds, dens_local, Tu, visc;
    double Retht;
    double densRe, densReduds, lamda, Fla;
    double theta_gss, theta0, theta, thetatmp, Feval, Feval0;

    densRe = dens_local / visc;
    densReduds = densRe * duds;

    theta_gss = Re_thetat(Tu, 1.0) / densRe;
    theta0 = theta_gss/100.0;
    theta = theta_gss * 100.0;
    lamda = stk::math::max(stk::math::min(0.10, densReduds * theta0 * theta0), -0.10);
    Fla = F_lamda(Tu, lamda);
    Feval0 = Re_thetat(Tu,Fla) - densRe * theta0;

    do iloop=1,itmax
    for (iloop = 1; iloop <= itmax) {
       lamda=stk::math::max(stk::math::min(densReduds * theta * theta,0.10), -0.10);
       Fla = F_lamda(Tu,lamda);
       Feval = Re_thetat(Tu, Fla) - densRe * theta;
       if(stk::math::abs(Feval - Feval0) < 1.e-3) break;
       thetatmp = theta;
       theta = theta - Feval*(theta - theta0)/(Feval - Feval0);
       theta0 = thetatmp;
       Feval0 = Feval;
    }

    Retht = Re_thetat(Tu, Fla);

    return Retht;
}

void
BLTRe0tNodeKernel::execute(
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
  const DblType sijMag    = sijMag_.get(node, 0);
  const DblType vortMag   = vortMag_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol      = dualNodalVolume_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

//
// needed for Re0t source term
//

// !!!!!!  need to be defined before this procedure is called: vortMag, dvels, velMag

//
// needed for Re0t source term
//

  DblType Reomega = density * sdr * minD * minD / visc;
  DblType Fwake = stk::math::exp(-1.0-10 * Reomega * Reomega);
  DblType delta = 350.0 * vortMag * minD * re0t * visc / (density * velMag * velMag);

  DblType farg = minD / delta;
  DblType crat = (gamint - 1.0/ceTwo)/(1.0-1.0/ceTwo);
  DblType f0t = stk::math::min(stk::math::max(fwake*exp(-farg*farg*farg*farg), 1.0 - crat*crat), 1.0);

  DblType tc  = 500.0 * visc / density / velMag / velMag;
  DblType Re0tcor = Secant_Re0tcor( dvels, density, Tu, visc);

  DblType Pre0t = c0t * density * (1.0 - f0t) / tc;

  rhs(0)    += Pre0t * Re0tcor * dVol;
  lhs(0, 0) += Pre0t * dVol;
}

} // namespace nalu
}  // sierra
