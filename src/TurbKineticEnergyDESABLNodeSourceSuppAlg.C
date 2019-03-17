/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <TurbKineticEnergyDESABLNodeSourceSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SupplementalAlgorithm.h>
#include <TimeIntegrator.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbKineticEnergyDESABLNodeSourceSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyDESABLNodeSourceSuppAlg::TurbKineticEnergyDESABLNodeSourceSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    tkeNp1_(NULL),
    sdrNp1_(NULL),
    densityNp1_(NULL),
    tvisc_(NULL),
    dudx_(NULL),
    dualNodalVolume_(NULL),
    maxLengthScale_(NULL),
    fOneBlend_(NULL),
    fABLBlending_(NULL),
    tkeProdLimitRatio_(realm_.get_turb_model_constant(TM_tkeProdLimitRatio)),	
    nDim_(realm.meta_data().spatial_dimension()),
    cDESke_(realm.get_turb_model_constant(TM_cDESke)),
    cDESkw_(realm.get_turb_model_constant(TM_cDESkw)),
    cEps_(realm_.get_turb_model_constant(TM_cEps))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *tke = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  tkeNp1_ = &(tke->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *sdr = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate");
  sdrNp1_ = &(sdr->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  maxLengthScale_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_max_length_scale");
  fOneBlend_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_f_one_blending");
  fABLBlending_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"abl_des_f_blending");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyDESABLNodeSourceSuppAlg::setup()
{
  // could extract user-based values
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyDESABLNodeSourceSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  const double tke        = *stk::mesh::field_data(*tkeNp1_, node );
  const double sdr        = *stk::mesh::field_data(*sdrNp1_, node );
  const double rho        = *stk::mesh::field_data(*densityNp1_, node );
  const double tvisc      = *stk::mesh::field_data(*tvisc_, node );
  const double *dudx      =  stk::mesh::field_data(*dudx_, node );
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node );
  const double maxLengthScale = *stk::mesh::field_data(*maxLengthScale_, node );
  const double fOneBlend = *stk::mesh::field_data(*fOneBlend_, node );
  const double fABLBlend = *stk::mesh::field_data(*fABLBlending_, node);


   double filter = std::pow(dualVolume, 1.0/nDim_);

  int nDim = nDim_;
  double Pk = 0.0;
  for ( int i = 0; i < nDim; ++i ) {
    const int offSet = nDim*i;
    for ( int j = 0; j < nDim; ++j ) {
      Pk += dudx[offSet+j]*(dudx[offSet+j] + dudx[nDim*j+i]);
    }
  }
  Pk *= tvisc;

  // Filter ksgs
  double Dksgs = cEps_*rho*std::pow(tke, 1.5)/filter;
 

  // blend cDES
  const double cDES = fOneBlend*cDESkw_ + (1.0-fOneBlend)*cDESke_;

  const double sqrtTke = std::sqrt(tke);
  const double lSST = sqrtTke/betaStar_/sdr;
  // prevent divide by zero (possible at resolved tke bcs = 0.0)
  const double lDES = std::max(1.0e-16, std::min(lSST, cDES*maxLengthScale));
  
  double Dkdes = rho*tke*sqrtTke/lDES;

  //ABLDES blending of source terms RHS and LHS  
  double Dk = Dksgs*fABLBlend + (1.0-fABLBlend)*Dkdes; 

  
  if ( Pk > tkeProdLimitRatio_*Dk )
    Pk = tkeProdLimitRatio_*Dk;
 
  double LinTerm = cEps_*rho*std::sqrt(tke)/filter*fABLBlend + (1-fABLBlend)*rho/lDES*sqrtTke;  
 
 
 
  rhs[0] += (Pk - Dk)*dualVolume;
  lhs[0] += 1.5*LinTerm*dualVolume;
}

} // namespace nalu
} // namespace Sierra
