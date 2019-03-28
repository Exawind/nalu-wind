/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <nso/ScalarNSOElemSuppAlgDep.h>
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
// ScalarNSOElemSuppAlgDep - NSO for scalar equation
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ScalarNSOElemSuppAlgDep::ScalarNSOElemSuppAlgDep(
  Realm &realm,
  ScalarFieldType *scalarQ,
  VectorFieldType *Gjq,
  ScalarFieldType *diffFluxCoeff,
  const double fourthFac,
  const double altResFac)
  : SupplementalAlgorithm(realm),
    bulkData_(&realm.bulk_data()),
    scalarQNm1_(NULL),
    scalarQN_(NULL),
    scalarQNp1_(NULL),
    densityNm1_(NULL),
    densityN_(NULL),
    densityNp1_(NULL),
    diffFluxCoeff_(diffFluxCoeff),
    velocityRTM_(NULL),
    Gjq_(Gjq),
    coordinates_(NULL),
    dt_(0.0),
    nDim_(realm_.spatialDimension_),
    gamma1_(0.0),
    gamma2_(0.0),
    gamma3_(0.0),
    Cupw_(0.1),
    small_(1.0e-16),
    fourthFac_(fourthFac),
    altResFac_(altResFac),
    om_altResFac_(1.0-altResFac),
    nonConservedForm_(0.0),
    useShiftedGradOp_(realm.get_shifted_grad_op(scalarQ->name()))
{
  // save off fields; for non-BDF2 gather in state N for Nm1 (gamma3_ will be zero)
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  scalarQNm1_ = realm_.number_of_states() == 2 ? &(scalarQ->field_of_state(stk::mesh::StateN)) : &(scalarQ->field_of_state(stk::mesh::StateNM1));
  scalarQN_ = &(scalarQ->field_of_state(stk::mesh::StateN));
  scalarQNp1_ = &(scalarQ->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNm1_ = realm_.number_of_states() == 2 ? &(density->field_of_state(stk::mesh::StateN)) : &(density->field_of_state(stk::mesh::StateNM1));
  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
 
  // check for mesh motion for proper velocity
  const bool meshMotion = realm_.does_mesh_move();
  if ( meshMotion )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // fixed size
  ws_dqdxScs_.resize(nDim_);
  ws_rhoVrtmScs_.resize(nDim_);
}

//--------------------------------------------------------------------------
//-------- elem_resize -----------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOElemSuppAlgDep::elem_resize(
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
  ws_qNm1_.resize(nodesPerElement);
  ws_qN_.resize(nodesPerElement);
  ws_qNp1_.resize(nodesPerElement);
  ws_rhoNp1_.resize(nodesPerElement);
  ws_rhoN_.resize(nodesPerElement);
  ws_rhoNm1_.resize(nodesPerElement);
  ws_velocityRTM_.resize(nDim_*nodesPerElement);
  ws_diffFluxCoeff_.resize(nodesPerElement);
  ws_Gjq_.resize(nDim_*nodesPerElement);
  ws_coordinates_.resize(nDim_*nodesPerElement);
  
  // compute shape function
  meSCS->shape_fcn(&ws_shape_function_[0]);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOElemSuppAlgDep::setup()
{
  dt_ = realm_.get_time_step();
  gamma1_ = realm_.get_gamma1();
  gamma2_ = realm_.get_gamma2();
  gamma3_ = realm_.get_gamma3();
}

//--------------------------------------------------------------------------
//-------- elem_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
ScalarNSOElemSuppAlgDep::elem_execute(
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
    ws_qNm1_[ni] = *stk::mesh::field_data(*scalarQNm1_, node);
    ws_qN_[ni] = *stk::mesh::field_data(*scalarQN_, node);
    ws_qNp1_[ni] = *stk::mesh::field_data(*scalarQNp1_, node);
 
    ws_rhoNm1_[ni] = *stk::mesh::field_data(*densityNm1_, node);
    ws_rhoN_[ni] = *stk::mesh::field_data(*densityN_, node);
    ws_rhoNp1_[ni] = *stk::mesh::field_data(*densityNp1_, node);
    
    ws_diffFluxCoeff_[ni] = *stk::mesh::field_data(*diffFluxCoeff_, node);

    // pointers to real data
    const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node );
    const double * coords = stk::mesh::field_data(*coordinates_, node );
    const double * Gjq    = stk::mesh::field_data(*Gjq_, node );

    // gather vectors
    const int offSet = ni*nDim_;
    for ( int j=0; j < nDim_; ++j ) {
      ws_coordinates_[offSet+j] = coords[j];
      ws_velocityRTM_[offSet+j] = vrtm[j];
      ws_Gjq_[offSet+j] = Gjq[j];
    }
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
   
    // zero out; scalar
    double qNm1Scs = 0.0;
    double qNScs = 0.0;
    double qNp1Scs = 0.0;
    double rhoNm1Scs = 0.0;
    double rhoNScs = 0.0;
    double rhoNp1Scs = 0.0;
    double dFdxAdv = 0.0;
    double dFdxDiff = 0.0;
    double dFdxCont = 0.0;
   
    // zero out vector
    for ( int i = 0; i < nDim_; ++i ) {
      ws_dqdxScs_[i] = 0.0;
      ws_rhoVrtmScs_[i] = 0.0;
    }
    
    // determine scs values of interest
    const int offSet = ip*nodesPerElement;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];

      // time term; scalar q
      qNm1Scs += r*ws_qNm1_[ic];
      qNScs += r*ws_qN_[ic];
      qNp1Scs += r*ws_qNp1_[ic];

      // time term, density
      rhoNm1Scs += r*ws_rhoNm1_[ic];
      rhoNScs += r*ws_rhoN_[ic];
      rhoNp1Scs += r*ws_rhoNp1_[ic];

      // compute scs derivatives and flux derivative
      const int offSetDnDx = nDim_*nodesPerElement*ip + ic*nDim_;
      const double qIC = ws_qNp1_[ic];
      const double rhoIC = ws_rhoNp1_[ic];
      const double diffFluxCoeffIC = ws_diffFluxCoeff_[ic];
      for ( int j = 0; j < nDim_; ++j ) {
        const double dnj = ws_dndx_[offSetDnDx+j];
        const double vrtmj = ws_velocityRTM_[ic*nDim_+j];
        ws_dqdxScs_[j] += qIC*dnj;
        ws_rhoVrtmScs_[j] += r*rhoIC*vrtmj;
        dFdxAdv += rhoIC*vrtmj*qIC*dnj;
        dFdxDiff += diffFluxCoeffIC*ws_Gjq_[ic*nDim_+j]*dnj;
        dFdxCont += rhoIC*vrtmj*dnj;
      }
    }
    
    // full continuity residual
    const double contRes = (gamma1_*rhoNp1Scs + gamma2_*rhoNScs + gamma3_*rhoNm1Scs)/dt_ + dFdxCont;

    // compute residual for NSO; linearized first
    double residualAlt = dFdxAdv - qNp1Scs*dFdxCont;
    for ( int j = 0; j < nDim_; ++j )
      residualAlt -= ws_rhoVrtmScs_[j]*ws_dqdxScs_[j];
    
    // compute residual for NSO; pde-based second
    const double time = (gamma1_*rhoNp1Scs*qNp1Scs + gamma2_*rhoNScs*qNScs + gamma3_*rhoNm1Scs*qNm1Scs)/dt_;
    const double residualPde = time + dFdxAdv - dFdxDiff - contRes*qNp1Scs*nonConservedForm_;

    // final form
    const double residual = residualAlt*altResFac_ + residualPde*om_altResFac_;

    // denominator for nu as well as terms for "upwind" nu
    double gUpperMagGradQ = 0.0;
    double rhoVrtmiGLowerRhoVrtmj = 0.0;
    for ( int i = 0; i < nDim_; ++i ) {
      const double dqdxScsi = ws_dqdxScs_[i];
      const double rhoVrtmi = ws_rhoVrtmScs_[i];
      for ( int j = 0; j < nDim_; ++j ) {
        gUpperMagGradQ += dqdxScsi*p_gUpper[i*nDim_+j]*ws_dqdxScs_[j];
        rhoVrtmiGLowerRhoVrtmj += rhoVrtmi*p_gLower[i*nDim_+j]*ws_rhoVrtmScs_[j];
      }
    }      
    
    // construct nu from residual
    const double nuResidual = std::sqrt((residual*residual)/(gUpperMagGradQ+small_));
    
    // construct nu from first-order-like approach; SNL-internal write-up (eq 209)
    // for now, only include advection as full set of terms is too diffuse
    const double nuFirstOrder = std::sqrt(rhoVrtmiGLowerRhoVrtmj);

    // limit based on first order; Cupw_ is a fudge factor similar to Guermond's approach
    const double nu = std::min(Cupw_*nuFirstOrder, nuResidual);
    
    double gijFac = 0.0;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];
      
      // save of some variables
      const double qIC = ws_qNp1_[ic];
      
      // NSO diffusion-like term; -nu*gUpper*(dQ/dxj - Gjq)*ai (residual below)
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
