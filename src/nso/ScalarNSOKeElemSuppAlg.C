/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <nso/ScalarNSOKeElemSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

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
// ScalarNSOKeElemSuppAlg - NSO for scalar equation
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ScalarNSOKeElemSuppAlg::ScalarNSOKeElemSuppAlg(
  Realm &realm,
  ScalarFieldType *scalarQ,
  VectorFieldType *Gjq,
  const double turbCoeff,
  const double fourthFac)
  : SupplementalAlgorithm(realm),
    bulkData_(&realm.bulk_data()),
    scalarQNp1_(NULL),
    densityNp1_(NULL),
    pressure_(NULL),
    velocityNp1_(NULL),
    velocityRTM_(NULL),
    Gjq_(Gjq),
    Gjp_(NULL),
    coordinates_(NULL),
    nDim_(realm_.spatialDimension_),
    Cupw_(0.1),
    small_(1.0e-16),
    turbCoeff_(turbCoeff),
    fourthFac_(fourthFac),
    useShiftedGradOp_(realm.get_shifted_grad_op(scalarQ->name()))
{
  // save off fields; for non-BDF2 gather in state N for Nm1 (gamma3_ will be zero)
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  scalarQNp1_ = &(scalarQ->field_of_state(stk::mesh::StateNP1));
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  densityNp1_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  velocityNp1_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");

  // check for mesh motion for proper velocity
  const bool meshMotion = realm_.does_mesh_move();
  if ( meshMotion )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");

  Gjp_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // fixed size
  ws_rhoVrtmScs_.resize(nDim_);
  ws_uNp1Scs_.resize(nDim_);
  ws_dpdxScs_.resize(nDim_);
  ws_GjpScs_.resize(nDim_);
  ws_dkedxScs_.resize(nDim_);
}

//--------------------------------------------------------------------------
//-------- elem_resize -----------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOKeElemSuppAlg::elem_resize(
  MasterElement *meSCS,
  MasterElement */*meSCV*/)
{
  const int nodesPerElement = meSCS->nodesPerElement_;
  const int numScsIp = meSCS->num_integration_points();

  // resize; geometry
  ws_scs_areav_.resize(numScsIp*nDim_);
  ws_dndx_.resize(nDim_*numScsIp*nodesPerElement);
  ws_deriv_.resize(nDim_*numScsIp*nodesPerElement);
  ws_det_j_.resize(numScsIp);
  ws_shape_function_.resize(numScsIp*nodesPerElement);
  ws_gUpper_.resize(nDim_*nDim_*numScsIp); // g^ij (covariant)
  ws_gLower_.resize(nDim_*nDim_*numScsIp); // g_ij (contravariat)

  // resize; fields
  ws_qNp1_.resize(nodesPerElement);
  ws_densityNp1_.resize(nodesPerElement);
  ws_pressure_.resize(nodesPerElement);
  ws_velocityNp1_.resize(nDim_*nodesPerElement);
  ws_velocityRTM_.resize(nDim_*nodesPerElement);
  ws_Gjq_.resize(nDim_*nodesPerElement);
  ws_Gjp_.resize(nDim_*nodesPerElement);
  ws_ke_.resize(nodesPerElement);
  ws_coordinates_.resize(nDim_*nodesPerElement);
 
  // compute shape function
  meSCS->shape_fcn(&ws_shape_function_[0]);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOKeElemSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- elem_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOKeElemSuppAlg::elem_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity element,
  MasterElement *meSCS,
  MasterElement */*meSCV*/)
{
  // details on this element topo
  const int nodesPerElement = meSCS->nodesPerElement_;
  const int numScsIp = meSCS->num_integration_points();
  const int *lrscv = meSCS->adjacentNodes();    
  
  // gather
  stk::mesh::Entity const *  node_rels = bulkData_->begin_nodes(element);
  int num_nodes = bulkData_->num_nodes(element);

  // sanity check on num nodes
  ThrowAssert( num_nodes == nodesPerElement );

  for ( int ni = 0; ni < num_nodes; ++ni ) {
    stk::mesh::Entity node = node_rels[ni];
    
    // gather scalars
    ws_qNp1_[ni] = *stk::mesh::field_data(*scalarQNp1_, node);
    ws_densityNp1_[ni] = *stk::mesh::field_data(*densityNp1_, node);
    ws_pressure_[ni] = *stk::mesh::field_data(*pressure_, node);

    // pointers to real data
    const double * uNp1   = stk::mesh::field_data(*velocityNp1_, node );
    const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node );
    const double * Gjq    = stk::mesh::field_data(*Gjq_, node );
    const double * Gjp    = stk::mesh::field_data(*Gjp_, node );
    const double * coords = stk::mesh::field_data(*coordinates_, node );
    
    // gather vectors
    double ke = 0.0;
    const int niNdim = ni*nDim_;
    for ( int j=0; j < nDim_; ++j ) {
      ws_velocityNp1_[niNdim+j] = uNp1[j];
      ws_velocityRTM_[niNdim+j] = vrtm[j];
      ws_Gjq_[niNdim+j] = Gjq[j];
      ws_Gjp_[niNdim+j] = Gjp[j];
      ws_coordinates_[niNdim+j] = coords[j];
      ke += uNp1[j]*uNp1[j]/2.0;
    }   
    ws_ke_[ni] = ke;
  }

  // compute geometry (AGAIN)...
  double scs_error = 0.0;
  meSCS->determinant(1, &ws_coordinates_[0], &ws_scs_areav_[0], &scs_error);
  
  // compute dndx (AGAIN)...
  if ( useShiftedGradOp_ )
    meSCS->shifted_grad_op(1, &ws_coordinates_[0], &ws_dndx_[0], &ws_deriv_[0], &ws_det_j_[0], &scs_error);
  else
    meSCS->grad_op(1, &ws_coordinates_[0], &ws_dndx_[0], &ws_deriv_[0], &ws_det_j_[0], &scs_error);
    
  // compute gij; requires a proper ws_deriv from above
  meSCS->gij(&ws_coordinates_[0], &ws_gUpper_[0], &ws_gLower_[0], &ws_deriv_[0]);

  for ( int ip = 0; ip < numScsIp; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv[2*ip];
    const int ir = lrscv[2*ip+1];

    // corresponding matrix rows
    const int rowL = il*nodesPerElement;
    const int rowR = ir*nodesPerElement;

    // pointer to gupperij and glowerij
    const double *p_gUpper = &ws_gUpper_[nDim_*nDim_*ip];
    const double *p_gLower = &ws_gLower_[nDim_*nDim_*ip];

    // zero out vector
    for ( int i = 0; i < nDim_; ++i ) {
      ws_rhoVrtmScs_[i] = 0.0;
      ws_uNp1Scs_[i] = 0.0;
      ws_dpdxScs_[i] = 0.0;
      ws_GjpScs_[i] = 0.0;
      ws_dkedxScs_[i] = 0.0;
    }
    double rhoScs = 0.0;

    // determine scs values of interest
    const int offSet = ip*nodesPerElement;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];

      // compute scs derivatives and flux derivative
      const int offSetDnDx = nDim_*nodesPerElement*ip + ic*nDim_;
      const double pressureIC = ws_pressure_[ic];
      const double rhoIC = ws_densityNp1_[ic];
      const double keIC = ws_ke_[ic];
      rhoScs += r*rhoIC;
      for ( int j = 0; j < nDim_; ++j ) {
        const double dnj = ws_dndx_[offSetDnDx+j];
        const double vrtm = ws_velocityRTM_[ic*nDim_+j];
        const double uNp1 = ws_velocityNp1_[ic*nDim_+j];
        const double Gjp = ws_Gjp_[ic*nDim_+j];
        ws_rhoVrtmScs_[j] += r*rhoIC*vrtm;
        ws_uNp1Scs_[j] += r*uNp1;
        ws_dpdxScs_[j] += pressureIC*dnj;
        ws_GjpScs_[j] += r*Gjp;
        ws_dkedxScs_[j] += keIC*dnj;
      }
    }

    // form ke residual (based on fine scale momentum residual used in Pstab)
    double keResidual = 0.0;
    for ( int j = 0; j < nDim_; ++j )
      keResidual += ws_uNp1Scs_[j]*(ws_dpdxScs_[j] - ws_GjpScs_[j])/rhoScs/2.0;
    
    // denominator for nu as well as terms for "upwind" nu
    double gUpperMagGradQ = 0.0;
    double rhoVrtmiGLowerRhoVrtmj = 0.0;
    for ( int i = 0; i < nDim_; ++i ) {
      const double dkedxScsi = ws_dkedxScs_[i];
      const double rhoVrtmi = ws_rhoVrtmScs_[i];
      for ( int j = 0; j < nDim_; ++j ) {
        gUpperMagGradQ += dkedxScsi*p_gUpper[i*nDim_+j]*ws_dkedxScs_[j];
        rhoVrtmiGLowerRhoVrtmj += rhoVrtmi*p_gLower[i*nDim_+j]*ws_rhoVrtmScs_[j];
      }
    }      
    
    // construct nu from ke residual
    const double nuResidual = rhoScs*std::sqrt((keResidual*keResidual)/(gUpperMagGradQ+small_));
    
    // construct nu from first-order-like approach; SNL-internal write-up (eq 209)
    // for now, only include advection as full set of terms is too diffuse
    const double nuFirstOrder = std::sqrt(rhoVrtmiGLowerRhoVrtmj);

    // limit based on first order; Cupw_ is a fudge factor similar to Guermond's approach
    const double nu = std::min(Cupw_*nuFirstOrder, nuResidual/turbCoeff_);
    
    double gijFac = 0.0;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];
      
      // save of some variables
      const double qIC = ws_qNp1_[ic];
      
      // NSO diffusion-like term; -nu*gUpper*(dq/dxj - Gjq)*ai (residual below)
      double lhsfac = 0.0;
      const int offSetDnDx = nDim_*nodesPerElement*ip + ic*nDim_;
      for ( int i = 0; i < nDim_; ++i ) {
        const double axi = ws_scs_areav_[ip*nDim_+i];
        for ( int j = 0; j < nDim_; ++j ) {
          const double dnxj = ws_dndx_[offSetDnDx+j];
          const double fac = p_gUpper[i*nDim_+j]*dnxj*axi;
          const double facGj = r*p_gUpper[i*nDim_+j]*ws_Gjq_[ic*nDim_+j]*axi;
          gijFac += fac*qIC - facGj*fourthFac_;
          lhsfac += -fac;
        }
      }
      
      lhs[rowL+ic] += nu*lhsfac;
      lhs[rowR+ic] -= nu*lhsfac;
    }
    
    // residual; left and right
    const double residualNSO = -nu*gijFac;
    rhs[il] -= residualNSO;
    rhs[ir] += residualNSO;
  }      
}
  
} // namespace nalu
} // namespace Sierra
