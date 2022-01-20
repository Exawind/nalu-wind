/*------------------------------------------------------------------------*/
/*  Copyright 2017 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTGammaM2015NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"
#include "NaluEnv.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

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

  // Update transition model constants
  caOne_ = realm.get_turb_model_constant(TM_caOne);
  caTwo_ = realm.get_turb_model_constant(TM_caTwo);
  ceOne_ = realm.get_turb_model_constant(TM_ceOne);
  ceTwo_ = realm.get_turb_model_constant(TM_ceTwo);
  timeStepCount = realm.get_time_step_count();
  maxStepCount  = realm.get_max_time_step_count();
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
  NALU_ALIGNED NodeKernelTraits::DblType dwalldistdx[NodeKernelTraits::NDimMax];
  NALU_ALIGNED NodeKernelTraits::DblType dndotvdx[NodeKernelTraits::NDimMax];

  const DblType tke       = tke_.get(node, 0);
  const DblType sdr       = sdr_.get(node, 0);
  const DblType gamint    = gamint_.get(node, 0);

  const DblType density   = density_.get(node, 0);
  const DblType visc      = visc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol      = dualNodalVolume_.get(node, 0);

  // define the wall normal vector (for now, hardwire to NASA TM case: z = wall norm direction)
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

  for (int d = 0; d < nDim_; d++) {
    coords[d] = coordinates_.get(node, d);
    vel[d] = velocityNp1_.get(node, d);
  }

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


  TuL = stk::math::min(81.6496580927726 * stk::math::sqrt(tke) / sdr / (minD + 1.0e-10), 100.0);
  lamda0L = -7.57e-3 * dvnn * minD * minD *density / visc + 0.0128;
  lamda0L = stk::math::min(stk::math::max(lamda0L, -1.0), 1.0);
  Re0c = Ctu1 + Ctu2 * stk::math::exp(-Ctu3 * TuL * FPG(lamda0L));
  Rev = density * minD * minD * sijMag / visc;
  fonset1 = Rev / 2.2 / Re0c;
  fonset2 = stk::math::min(fonset1, 2.0);
  rt = density*tke/sdr/visc;
  fonset3 = stk::math::max(1.0 - 0.0233236151603499 * rt * rt * rt, 0.0);
  fonset = stk::math::max(fonset2 - fonset3, 0.0);
  fturb = stk::math::exp(-rt * rt * rt * rt / 16.0);

  DblType Pgamma =   flength * density * sijMag * fonset * gamint * ( 1.0 - gamint );
  DblType Dgamma = - caTwo_ * density * vortMag * fturb  * gamint * ( ceTwo_ * gamint - 1.0 );

  DblType PgammaDir =   flength * density * sijMag * fonset * ( 1.0 - 2.0 * gamint );
  DblType DgammaDir = - caTwo_ * density * vortMag * fturb  * ( 2.0 * ceTwo_ * gamint - 1.0 );

    const double dx   = stk::math::abs(coords[0] - 0.50);
    const double dy   = stk::math::abs(coords[1] + 0.50);

#if 0
    //printf("y = %.8E dy = %.8E\n", coords[1], dy);

    if (dy < 0.1 && timeStepCount == 100) {
      NaluEnv::self().naluOutputP0() <<
        timeStepCount << " " << coords[0] << " " << coords[1] << " " << coords[2] << " " << vel[0] << " " << 
        vel[2] << " " << gamint << " " << tke << " " << sdr << " " << minD << " " << 
        dvnn << " " << TuL << " " << lamda0L << " " << Re0c << " " << Rev << " " << 
        fonset1 << " " << fonset2 << " " << fonset3 << " " << fonset << " " << fturb << " " << 
        rt << " " << sijMag << " " << vortMag << " " << Pgamma << " " << Dgamma << " " << 
        Pgamma-Dgamma << std::endl;

   if (dx < 0.08) {
      NaluEnv::self().naluOutputP0() <<
        timeStepCount+0.12345678 << " " << coords[0] << " " << coords[1] << " " << coords[2] << " " << vel[0] << " " <<
        vel[2] << " " << gamint << " " << tke << " " << sdr << " " << minD << " " <<
        dvnn << " " << TuL << " " << lamda0L << " " << Re0c << " " << Rev << " " <<
        fonset1 << " " << fonset2 << " " << fonset3 << " " << fonset << " " << fturb << " " <<
        rt << " " << sijMag << " " << vortMag << " " << Pgamma << " " << Dgamma << " " <<
        Pgamma-Dgamma << std::endl;
   }

   }
#endif

  rhs(0) += (Pgamma + Dgamma) * dVol;
  lhs(0, 0) -= (PgammaDir + DgammaDir) * dVol;

}

} // namespace nalu
}  // sierra
