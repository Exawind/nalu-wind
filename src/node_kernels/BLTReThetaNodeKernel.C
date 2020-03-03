/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTReThetaNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

BLTReThetaNodeKernel::BLTReThetaNodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<BLTReThetaNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    re0tID_(get_field_ordinal(meta, "Retheta_Eq_transition")),
    velocityNp1ID_(get_field_ordinal(meta, "velocity")),
    nDim_(meta.spatial_dimension())
{}

void
BLTReThetaNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  dkdx_            = fieldMgr.get_field<double>(dkdxID_);
  dwdx_            = fieldMgr.get_field<double>(dwdxID_);
  minD_            = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  re0t_            = fieldMgr.get_field<double>(re0tID_);
  velocityNp1_     = fieldMgr.get_field<double>(velocityNp1ID_);

  // Update turbulence model constants
  c0t_     = realm.get_turb_model_constant(TM_c0t);
  ceTwo_   = realm.get_turb_model_constant(TM_ceTwo);
}

double
BLTReThetaNodeKernel::Re_thetat(const double& Tu, const double& Fla)
{
    using DblType = NodeKernelTraits::DblType;

    DblType Re0t;

    if (Tu <= 1.3) {  
       Re0t = ( 1173.51 - 589.428 * Tu + 0.2196 / Tu / Tu) * Fla;
    }
    else {
       Re0t = 331.5 * stk::math::pow(Tu - 0.5658, -0.671) * Fla;
    }

    Re0t = stk::math::max(Re0t, 20.0);

    return Re0t;
}

double
BLTReThetaNodeKernel::F_lamda(const double& Tu, const double& lamda)
{
    using DblType = NodeKernelTraits::DblType;

    DblType Fla;

    if ( lamda <= 0.0) {  
       Fla = 1.0 + (12.986 * lamda + 123.66 * lamda*lamda + 405.689 * lamda*lamda*lamda) * stk::math::exp(-stk::math::pow(Tu/1.5, 1.5) );
    }
    else {
       Fla = 1.0 + 0.275 * (1.0-stk::math::exp(-35.0 * lamda)) * stk::math::exp(-2.0*Tu);
    }

    return Fla;
}

double
BLTReThetaNodeKernel::Secant_Re0tcor(const double& duds, const double& dens_local, const double& Tu, const double& visc)
{
    using DblType = NodeKernelTraits::DblType;

    const int itmax = 10;
    int iloop;

    DblType Retht = 0.0;
    DblType densRe = 0.0;
    DblType densReduds = 0.0;
    DblType lamda = 0.0;
    DblType Fla = 0.0;

    DblType theta_gss = 0.0;
    DblType theta0 = 0.0;
    DblType theta = 0.0;
    DblType thetatmp = 0.0;
    DblType Feval = 0.0;
    DblType Feval0 = 0.0;

    densRe = dens_local / visc;
    densReduds = densRe * duds;

    theta_gss = Re_thetat(Tu, 1.0) / densRe;
    theta0 = theta_gss/100.0;
    theta = theta_gss * 100.0;
    lamda = stk::math::max(stk::math::min(0.10, densReduds * theta0 * theta0), - 0.10);
    Fla = F_lamda(Tu, lamda);
    Feval0 = Re_thetat(Tu,Fla) - densRe * theta0;

    for (iloop = 1; iloop <= itmax; ++iloop) {
       lamda=stk::math::max(stk::math::min(densReduds * theta * theta,0.10), - 0.10);
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
BLTReThetaNodeKernel::execute(
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
  const DblType visc      = visc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol      = dualNodalVolume_.get(node, 0);

  NALU_ALIGNED NodeKernelTraits::DblType vel[NodeKernelTraits::NDimMax];
//
// needed for Re0t source term
//

  DblType velMag = 0.0;
  DblType Reomega = 0.0;
  DblType Fwake = 0.0;
  DblType delta = 0.0;
  DblType farg = 0.0;
  DblType crat = 0.0;
  DblType f0t  = 0.0;
  DblType tc   = 0.0;
  DblType dudx = 0.0;
  DblType dudy = 0.0;
  DblType dudz = 0.0;

  DblType dvdx = 0.0;
  DblType dvdy = 0.0;
  DblType dvdz = 0.0;

  DblType dwdx = 0.0;
  DblType dwdy = 0.0;
  DblType dwdz = 0.0;

  DblType dvelx = 0.0;
  DblType dvely = 0.0;
  DblType dvelz = 0.0;
  DblType dvels = 0.0;
  DblType Tu    = 0.0;

  DblType Re0tcor = 0.0;
  DblType Pre0t = 0.0;
//

  for (int i=0; i < NodeKernelTraits::NDimMax; ++i) {
    vel[i] = velocityNp1_.get(node, i);
  }
  
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

  velMag = stk::math::sqrt( vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2] ) + 1.e-14;
  Reomega = density * sdr * minD * minD / visc;
  Fwake = stk::math::exp(-1.0e-10 * Reomega * Reomega);
  delta = 350.0 * vortMag * minD * re0t * visc / (density * velMag * velMag);
  farg = minD / delta;
  crat = (gamint - 1.0/ceTwo_)/(1.0-1.0/ceTwo_);
  f0t = stk::math::min(stk::math::max(Fwake*exp(-farg*farg*farg*farg), 1.0 - crat*crat), 1.0);
  tc  = 500.0 * visc / density / velMag / velMag;

  dudx = dudx_.get(node, nDim_*0 + 0);
  dudy = dudx_.get(node, nDim_*0 + 1);
  dudz = dudx_.get(node, nDim_*0 + 2);

  dvdx = dudx_.get(node, nDim_*1 + 0);
  dvdy = dudx_.get(node, nDim_*1 + 1);
  dvdz = dudx_.get(node, nDim_*1 + 2);

  dwdx = dudx_.get(node, nDim_*2 + 0);
  dwdy = dudx_.get(node, nDim_*2 + 1);
  dwdz = dudx_.get(node, nDim_*2 + 2);

  dvelx = vel[0]*dudx+vel[1]*dvdx+vel[2]*dwdx;
  dvely = vel[0]*dudy+vel[1]*dvdy+vel[2]*dwdy;
  dvelz = vel[0]*dudz+vel[1]*dvdz+vel[2]*dwdz;

  dvels = 1.0/(velMag * velMag) * (vel[0]*dvelx+vel[1]*dvely+vel[2]*dvelz);
  Tu = stk::math::max(81.6496580927726e0 * stk::math::sqrt(tke)/velMag, 0.027e0);

  Re0tcor = Secant_Re0tcor( dvels, density, Tu, visc);

  Pre0t = c0t_ * density * (1.0 - f0t) / tc;

  rhs(0)    += Pre0t * ( Re0tcor - re0t ) * dVol;
  lhs(0, 0) += Pre0t * dVol;
}

} // namespace nalu
}  // sierra
