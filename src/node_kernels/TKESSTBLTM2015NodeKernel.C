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
    coordinatesID_(get_field_ordinal(meta, "coordinates")),
    velocityNp1ID_(get_field_ordinal(meta, "velocity")),
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
  coordinates_     = fieldMgr.get_field<double>(coordinatesID_);
  velocityNp1_     = fieldMgr.get_field<double>(velocityNp1ID_);
  dt_              = realm.get_time_step();

  const std::string dofName = "turbulent_ke";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  FILE * fp;
  fp = std::fopen ("tkeFreestream.dat", "r");
  std::fscanf(fp,"%lf\n",&tkeFreestream);
//  std::printf("Node Kernel Inlet K value = %.12E\n", tkeFreestream);
  std::fclose(fp);

  // Update turbulence model constants
  betaStar_          = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  c0t_               = realm.get_turb_model_constant(TM_c0t);
}

void TKESSTBLTM2015NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // Modify Production and destruction TKE source terms with gamma for transition
  // See https://turbmodels.larc.nasa.gov/sst.html for details

  NALU_ALIGNED NodeKernelTraits::DblType coords[NodeKernelTraits::NDimMax]; // coordinates
  NALU_ALIGNED NodeKernelTraits::DblType vel[NodeKernelTraits::NDimMax];

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
  DblType velMag2   = 0.0;
  DblType Pk = 0.0;
  DblType Pklim = 0.0;
  DblType tkeForcing = 0.0;
  DblType tkeZero = 1.0;
  DblType tc = 0.0;
//  DblType t1 = 0.0;
//  DblType t2 = 0.0;
//  DblType t3 = 0.0;
//  DblType t4 = 0.0;

/////////////////////
//  inflowData.tke_ = node["turbulent_ke"].as<sierra::nalu::TurbKinEnergy>();
//
//  bool convert<sierra::nalu::TurbKinEnergy>::decode(const Node& node,
//    sierra::nalu::TurbKinEnergy& tke)
//  {
//    if (!node.IsScalar())
//    {
//      return false;
//    }
//
//    tke.turbKinEnergy_ = node.as<double>();
//
//    return true;
//  }
///////////////
  for (int d = 0; d < nDim_; d++) {
    coords[d] = coordinates_.get(node, d);
    vel[d] = velocityNp1_.get(node, d);
  }

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

  sijMag = stk::math::sqrt(sijMag);
  vortMag = stk::math::sqrt(vortMag);
  velMag2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2] + 1.e-14;

  DblType Rev = density * minD*minD * sijMag / visc;
  DblType Fonlim = stk::math::min(stk::math::max(Rev / 2.20 / Re0clim - 1.0, 0.0), 3.0);

//   const DblType gamint_debug = gamint;
   const DblType gamint_debug = gamint;

  if (coords[0] < -0.04) {
    tc = 500.0 * visc / density / velMag2; 
    tkeForcing = c0t_ * density * (tkeFreestream - tke) / tc;
    rhs(0) += tkeForcing * dVol;
    lhs(0, 0) += c0t_ * density * dVol/ tc;
  }
  else {
//  Pk = tvisc * sijMag * vortMag; // Pk based on Kato-Launder formulation. Recommended in Menter (2015) to avoid excessive levels of TKE in stagnation regions
    Pk *= tvisc*gamint_debug; // based on standard SST model
    Pklim = 5.0 * Ck_BLT * stk::math::max(gamint_debug - 0.20, 0.0) * (1.0 - gamint_debug) * Fonlim * stk::math::max(3.0 * CSEP * visc - tvisc, 0.0) * sijMag * vortMag;
    DblType Dk = betaStar_ * density * sdr * tke * stk::math::max(gamint_debug,0.1);
  // Clip production term
    Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk); // ???????
    rhs(0) += (Pk + Pklim - Dk) * dVol;
    lhs(0, 0) += betaStar_ * density * sdr * stk::math::max(gamint_debug,0.1) * dVol;
  }

//  if (stk::math::abs(Pklim) > 1.0e-10) {
//    t1 = stk::math::max(gamint_debug - 0.20, 0.0);
//    t2 = (1.0 - gamint_debug);
//    t3 = Fonlim;
//    t4 = stk::math::max(3.0 * CSEP * visc - tvisc, 0.0);
//
//    std::cout << "Pklim is non-zero: " << Pklim << " minD = " << minD << " t1 = " << t1 << " t2 = " << t2 << " t3 = " << t3 << " t4 = " << t4 << " sijMag = " << sijMag << " vortMag = " << vortMag << std::endl;
//  }
//  DblType Dk = betaStar_ * density * sdr * tke * stk::math::max(gamint_debug,0.1);

  // Clip production term//  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk); // ???????
  //
//  rhs(0) += (tkeZero * (Pk + Pklim - Dk) + tkeForcing) * dVol;
//  lhs(0, 0) += tkeZero * betaStar_ * density * sdr * stk::math::max(gamint_debug,0.1) * dVol;
}

}  // nalu
}  // sierra
