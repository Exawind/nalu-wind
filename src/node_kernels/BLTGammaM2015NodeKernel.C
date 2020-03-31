/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTGammaM2015NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

BLTGammaM2015NodeKernel::BLTGammaM2015NodeKernel(
  const stk::mesh::MetaData& meta
) : NGPNodeKernel<BLTGammaM2015NodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    coordinatesID_(get_field_ordinal(meta, "coordinates")),
    velocityNp1ID_(get_field_ordinal(meta, "velocity")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    gammaprodID_(get_field_ordinal(meta, "gamma_production")),
    gammasinkID_(get_field_ordinal(meta, "gamma_sink")),
    gammarethID_(get_field_ordinal(meta, "gamma_reth")),
    nDim_(meta.spatial_dimension())
{}

void
BLTGammaM2015NodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_             = fieldMgr.get_field<double>(tkeID_);
  sdr_             = fieldMgr.get_field<double>(sdrID_);
  density_         = fieldMgr.get_field<double>(densityID_);
  visc_            = fieldMgr.get_field<double>(viscID_);
  dudx_            = fieldMgr.get_field<double>(dudxID_);
  minD_            = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_     = fieldMgr.get_field<double>(coordinatesID_);
  velocityNp1_     = fieldMgr.get_field<double>(velocityNp1ID_);
  gamint_          = fieldMgr.get_field<double>(gamintID_);
  gammaprod_      = fieldMgr.get_field<double>(gammaprodID_);
  gammasink_      = fieldMgr.get_field<double>(gammasinkID_);
  gammareth_      = fieldMgr.get_field<double>(gammarethID_);



  // Update transition model constants
  caOne_ = realm.get_turb_model_constant(TM_caOne);
  caTwo_ = realm.get_turb_model_constant(TM_caTwo);
  ceOne_ = realm.get_turb_model_constant(TM_ceOne);
  ceTwo_ = realm.get_turb_model_constant(TM_ceTwo);
  timeStepCount = realm.get_time_step_count();
}

double
BLTGammaM2015NodeKernel::BLVel(const double& density, const double& visc, const double& uinf, const double& cX, const double& cZ)
{   
  using DblType = NodeKernelTraits::DblType;

  DblType Unorm;
  DblType out;
  DblType arg1;
  DblType xtmp;
  DblType rex;

   xtmp = stk::math::max(cX , 0.001);
   rex = density * uinf * xtmp / visc;
   arg1 = stk::math::pow(rex, 1.0/7.0) / 0.16 / xtmp;
   Unorm = std::pow(cZ * arg1, 1.0/7.0);
   out = uinf * std::min(Unorm, 1.0);
  
  return out;
}


double
BLTGammaM2015NodeKernel::FPG(const double& lamda0L )
{
    using DblType = NodeKernelTraits::DblType;

    DblType out; // this is the result of this calculation
    DblType CPG1=14.68;
    DblType CPG2=-7.34;
    DblType CPG3=0.0;

    DblType CPG1_lim=1.5;
    DblType CPG2_lim=3.0;

    if(lamda0L >= 0.0) {
       out = stk::math::min(1.0 + CPG1 * lamda0L, CPG1_lim);
    }
    else {
       out = stk::math::min(1.0 + CPG2 * lamda0L + CPG3 * stk::math::min(lamda0L + 0.0681, 0.0), CPG2_lim);
    }

    out=stk::math::max(out, 0.0);

  return out;
}

void
BLTGammaM2015NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  NALU_ALIGNED NodeKernelTraits::DblType coords[NodeKernelTraits::NDimMax]; // coordinates
  NALU_ALIGNED NodeKernelTraits::DblType vel[NodeKernelTraits::NDimMax];

  const DblType tke       = tke_.get(node, 0);
  const DblType sdr       = sdr_.get(node, 0);
  const DblType gamint    = gamint_.get(node, 0);

//  DblType gammaprod = gammaprod_.get(node, 0);
//  DblType gammasink = gammasink_.get(node, 0);
//  DblType gammareth = gammareth_.get(node, 0);

  const DblType density   = density_.get(node, 0);
  const DblType visc      = visc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol      = dualNodalVolume_.get(node, 0);

  // define the wall normal vector (for now, hardwire to NASA TM case: z = wall norm direction)
  int wallnorm_dir = 2;

  DblType Re0c = 0.0;
  DblType flength = 100.0;
  DblType Rev = 0.0;
  DblType rt = 0.0;
  DblType dvnn = 0.0;
  DblType TuL = 0.0;
  DblType lamda0L = 0.0;
  
  DblType fonset  = 0.0;
  DblType fonset1 = 0.0;
  DblType fonset2 = 0.0;
  DblType fonset3 = 0.0;
  DblType fturb = 0.0;

  DblType sijMag = 0.0;
  DblType vortMag = 0.0;

  DblType Ctu1=100.;
  DblType Ctu2=1000.;
  DblType Ctu3=1.0;

  DblType le_fact = 1.0;
  //double uinf=19.8;
  //double dux = 0.0;
  //double duz = 0.0;
  //double dwx = 0.0;
  //double dwz = 0.0;
  //double eps = 1.0e-5;

  //double xp = 0.0;
  //double xc = 0.0;
  //double xm = 0.0;
  //double zp = 0.0;
  //double zc = 0.0;
  //double zm = 0.0;

  //double uxc = 0.0;
  //double uxp = 0.0;
  //double uxm = 0.0;
  //double uzp = 0.0;
  //double uzm = 0.0;


  for (int d = 0; d < nDim_; d++) {
    coords[d] = coordinates_.get(node, d);
    vel[d] = velocityNp1_.get(node, d);
  }

// Debug: fix velocity &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //xtmp = stk::math::max(coords[0] , 0.001);
  //rex = density * uinf * xtmp / visc;
  //arg1 = stk::math::pow(rex, 1.0/7.0) / 0.16 / xtmp;

  //vel[0] = uinf * stk::math::pow(coords[2] * arg1, 1.0/7.0);
  //vel[1] = 0.0;
  //vel[2] = 0.0;

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

//  xp = coords[0] + eps;
//  xc = coords[0];
//  xm = coords[0] - eps;

//  zp = coords[2] + eps;
//  zc = coords[2];
//  zm = coords[2] - eps;
//  zm = stk::math::max(zm, 0.0);

//  uxp = BLVel(density, visc, uinf, xp, zc);
//  uxc = BLVel(density, visc, uinf, xc, zc);
//  uxm = BLVel(density, visc, uinf, xm, zc);

//  uzp = BLVel(density, visc, uinf, xc, zp);
//  uzm = BLVel(density, visc, uinf, xc, zm);

//  vel[0] = uxc;
//  vel[1] = 0.0;
//  vel[2] = 0.0;

//  dux = (uxp - uxm)/(xp-xm);
//  duz = (uzp - uzm)/(zp-zm);

//  sijMag =  stk::math::sqrt( (duz+dwx)*(duz+dwx) + 2.0*dux*dux + 2.0*dwz*dwz );
//  vortMag = stk::math::sqrt( (duz-dwx)*(duz-dwx) );

// dvnn = wall normal derivative of wall normal velocity (for now, hardwire to NASA TM case: z = wall norm direction)
//  dvnn = dudx_.get(node, nDim_ * wallnorm_dir + wallnorm_dir);
// %%%%%%%%%%%%%%%%%%%%%%%%%%
  //!!!!!!!!!!!!!! Just for debug, turn off pressure gradient
  dvnn = 0.0;
// %%%%%%%%%%%%%%%%%%%%%%%%%%

  TuL = stk::math::min(81.6496580927726 * stk::math::sqrt(tke) / sdr / (minD + 1.0e-10), 100.0);
// %%%%%%%%%%%%%%%%%%%%%%%%%%
  //lamda0L = -7.57e-3 * dvnn * minD * minD *density / visc + 0.0128;
  //// %%%%%%%%%%%%%%%%%%%%%%%%%%
  lamda0L = 0.0128;
  lamda0L = stk::math::min(stk::math::max(lamda0L, -1.0), 1.0);
  Re0c = Ctu1 + Ctu2 * stk::math::exp(-Ctu3 * TuL * FPG(lamda0L));
  Rev = density * minD * minD * sijMag / visc;
  fonset1 = Rev / 2.2 / Re0c;
  fonset2 = stk::math::min(fonset1, 2.0);
  rt = density*tke/sdr/visc;
  fonset3 = stk::math::max(1.0 - 0.0233236151603499 * rt * rt * rt, 0.0);
  fonset = stk::math::max(fonset2 - fonset3, 0.0);
  fturb = stk::math::exp(-rt * rt * rt * rt / 16.0);

  DblType Pgamma = flength * density * sijMag * gamint * ( 1.0 - gamint ) * fonset;
  DblType Dgamma = caTwo_ * density * vortMag * gamint * fturb * ( ceTwo_ * gamint - 1.0 );


  DblType PgammaDeriv = -flength * density * sijMag * fonset * (1.0 - 2.0 * gamint );
  DblType DgammaDeriv = caTwo_ * density * vortMag * fturb * ( 2.0*ceTwo_ * gamint - 1.0 );

//  gammaprod_ = Pgamma;
//  gammasink_ = Dgamma;
//  gammareth_ = Re0c;
//  const double dy   = stk::math::abs(coords[1] + 0.50);

  const double dxle   = stk::math::abs(coords[0] - 0.00);
  const double dzle   = stk::math::abs(coords[2] - 0.00);
  const double dist = stk::math::sqrt(dxle*dxle + dzle*dzle) ;

  if (dist <= 0.005) {
    le_fact = dist/0.005;
  }

    //Pgamma = 0.0;
    //Dgamma = 0.0;
    //PgammaDeriv = 0.0;
    //DgammaDeriv = 0.0;

//    if (dy < 0.1 && timeStepCount <= 3) {
//      std::printf("%i %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E %.12E\n", 
//        timeStepCount, coords[0], coords[1], coords[2], vel[0], vel[2], gamint, tke, sdr, minD, dvnn, TuL, lamda0L, Re0c, Rev, fonset1, fonset2, fonset3, fonset, fturb, 
//        rt, sijMag, vortMag, Pgamma, Dgamma, Pgamma-Dgamma);
//    }

  rhs(0) += le_fact*(Pgamma - Dgamma) * dVol;
  lhs(0, 0) -= 0.0*le_fact*(PgammaDeriv - DgammaDeriv) * dVol;

}

} // namespace nalu
}  // sierra
