/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MomentumGclSrcNodeSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>
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
// MomentumGclSrcNodeSuppAlg - GCL
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumGclSrcNodeSuppAlg::MomentumGclSrcNodeSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    velocityNp1_(NULL),
    densityNp1_(NULL),
    divV_(NULL),
    dualNdVolN_(NULL),
    dualNdVolNp1_(NULL),
    nDim_(1),
    gamma1_(0.0),
    gamma2_(0.0),
    gamma3_(0.0)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  VectorFieldType *velocity = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocityNp1_ = &(velocity->field_of_state(stk::mesh::StateNP1));

  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));

  divV_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "div_mesh_velocity");

  ScalarFieldType *dualNdVol = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  dualNdVolNm1_ = realm_.number_of_states() == 2 ? &(dualNdVol->field_of_state(stk::mesh::StateN)) : &(dualNdVol->field_of_state(stk::mesh::StateNM1));
  dualNdVolN_ = &(dualNdVol->field_of_state(stk::mesh::StateN));
  dualNdVolNp1_ = &(dualNdVol->field_of_state(stk::mesh::StateNP1));

  nDim_ = meta_data.spatial_dimension();
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumGclSrcNodeSuppAlg::setup()
{
  gamma1_ = realm_.get_gamma1();
  gamma2_ = realm_.get_gamma2();
  gamma3_ = realm_.get_gamma3();

  dt_ = realm_.timeIntegrator_->get_time_step();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumGclSrcNodeSuppAlg::node_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity node)
{
  // rhs-= rho*u*div(v)
  const double *uNp1 = stk::mesh::field_data(*velocityNp1_, node );
  const double rhoNp1 = *stk::mesh::field_data(*densityNp1_, node );
  const double divV = *stk::mesh::field_data(*divV_, node );
  const double dualVolumeNm1 = *stk::mesh::field_data(*dualNdVolNm1_, node );
  const double dualVolumeN = *stk::mesh::field_data(*dualNdVolN_, node );
  const double dualVolumeNp1 = *stk::mesh::field_data(*dualNdVolNp1_, node );

  double volRate = (gamma1_*dualVolumeNp1 + gamma2_*dualVolumeN + gamma3_*dualVolumeNm1) / dt_ / dualVolumeNp1;

  // the term divV comes from the Reynold's transport theorem for moving bodies with changing volume
  // the term (volRate-divV) is the GCL law which presents non-zero errors in a discretized setting
  const double fac = rhoNp1*(divV - (volRate-divV))*dualVolumeNp1;
  for ( int i = 0; i < nDim_; ++i ) {
    rhs[i] -= fac*uNp1[i];
  }
}

} // namespace nalu
} // namespace Sierra
