/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <SupplementalAlgorithm.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumElemSolverAlgorithm - add LHS/RHS for uvw momentum
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemSolverAlgorithm::AssembleMomentumElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm_.get_divU()),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    velocity_(NULL),
    coordinates_(NULL),
    dudx_(NULL),
    density_(NULL),
    viscosity_(NULL),
    massFlowRate_(NULL),
    pecletFunction_(NULL)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  const std::string viscName = realm.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  massFlowRate_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function<double>(velocity_->name());

  /* Notes:

  Matrix layout is in row major. For a npe = 4 (quad) and nDim = 2:

  RHS = (resUx0, resUy0, resUx1, resUy1, resUx2, resUy2, resUx3, resUy3)

  where Uik = velocity_i_node_k

  The LHS is, therefore,

  row 0: d/dUx0(ResUx0), d/dUy0(ResUx0), ., ., ., .,  d/dUx3(ResUx0), d/dUy3(ResUx0)
  row 1: d/dUx0(ResUy0), d/dUy0(ResUy0), ., ., ., .,  d/dUx3(ResUy0), d/dUy3(ResUy0)

  */
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemSolverAlgorithm::~AssembleMomentumElemSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = "velocity";
  const double alpha = realm_.get_alpha_factor(dofName);
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);
  const bool useShiftedGradOp = realm_.get_shifted_grad_op(dofName);
  const bool skewSymmetric = realm_.get_skew_symmetric(dofName);

  const double relaxFacU = realm_.solutionOptions_->get_relaxation_factor(dofName);

  // one minus flavor..
  const double om_alpha = 1.0-alpha;
  const double om_alphaUpw = 1.0-alphaUpw;

  // space for LHS/RHS; nodesPerElem*nDim*nodesPerElem*nDim and nodesPerElem*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // supplemental algorithm setup
  const size_t supplementalAlgSize = supplementalAlg_.size();
  for ( size_t i = 0; i < supplementalAlgSize; ++i )
    supplementalAlg_[i]->setup();

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_vrtm;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_dudx;
  std::vector<double> ws_densityNp1;
  std::vector<double> ws_viscosity;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_shape_function;
  std::vector<double> ws_adv_shape_function;

  // ip values
  std::vector<double>uIp(nDim);

  // extrapolated value from the L/R direction 
  std::vector<double>uIpL(nDim);
  std::vector<double>uIpR(nDim);
  // limiter values from the L/R direction, 0:1
  std::vector<double> limitL(nDim,1.0); 
  std::vector<double> limitR(nDim,1.0);
  // extrapolated gradient from L/R direction
  std::vector<double> duL(nDim);
  std::vector<double> duR(nDim);
 
  // coords
  std::vector<double>coordIp(nDim);

  // pointers for fast access
  double *p_uIp = &uIp[0];
  double *p_uIpL = &uIpL[0];
  double *p_uIpR = &uIpR[0];
  double *p_limitL = &limitL[0];
  double *p_limitR = &limitR[0];
  double *p_duL = &duL[0];
  double *p_duR = &duR[0];
  double *p_coordIp = &coordIp[0];

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->num_integration_points();
    const int *lrscv = meSCS->adjacentNodes();

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
    const int rhsSize = nodesPerElement*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // algorithm related
    ws_velocityNp1.resize(nodesPerElement*nDim);
    ws_vrtm.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_dudx.resize(nodesPerElement*nDim*nDim);
    ws_densityNp1.resize(nodesPerElement);
    ws_viscosity.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_dndx.resize(nDim*numScsIp*nodesPerElement);
    ws_deriv.resize(nDim*numScsIp*nodesPerElement);
    ws_det_j.resize(numScsIp);
    ws_shape_function.resize(numScsIp*nodesPerElement);
    if ( skewSymmetric )
      ws_adv_shape_function.resize(numScsIp*nodesPerElement);

    // pointer to lhs/rhs
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_vrtm = &ws_vrtm[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_dudx = &ws_dudx[0];
    double *p_densityNp1 = &ws_densityNp1[0];
    double *p_viscosity = &ws_viscosity[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_dndx = &ws_dndx[0];
    double *p_shape_function = &ws_shape_function[0];
    double *p_adv_shape_function = skewSymmetric ? &ws_adv_shape_function[0] : &ws_shape_function[0];
    
    // extract shape function
    meSCS->shape_fcn(&p_shape_function[0]);
    if ( skewSymmetric )
      meSCS->shifted_shape_fcn(&p_adv_shape_function[0]);
   
    // resize possible supplemental element alg
    for ( size_t i = 0; i < supplementalAlgSize; ++i )
      supplementalAlg_[i]->elem_resize(meSCS, meSCV);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get elem
      stk::mesh::Entity elem = b[k];

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // ip data for this element; scs and scv
      const double *mdot = stk::mesh::field_data(*massFlowRate_, b, k );

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // set connected nodes
        connected_nodes[ni] = node;

        // pointers to real data
        const double * uNp1   =  stk::mesh::field_data(velocityNp1, node);
        const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node);
        const double * coords =  stk::mesh::field_data(*coordinates_, node);
        const double * du     =  stk::mesh::field_data(*dudx_, node);
        const double rhoNp1   = *stk::mesh::field_data(densityNp1, node);
        const double mu       = *stk::mesh::field_data(*viscosity_, node);

        // gather scalars
        p_densityNp1[ni] = rhoNp1;
        p_viscosity[ni] = mu;

        // gather vectors
        const int niNdim = ni*nDim;

        // row for p_dudx
        const int row_p_dudx = niNdim*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_velocityNp1[niNdim+i] = uNp1[i];
          p_vrtm[niNdim+i] = vrtm[i];
          p_coordinates[niNdim+i] = coords[i];
          // gather tensor
          const int row_dudx = i*nDim;
          for ( int j=0; j < nDim; ++j ) {
            p_dudx[row_p_dudx+row_dudx+j] = du[row_dudx+j];
          }
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // compute dndx
      if ( useShiftedGradOp )
        meSCS->shifted_grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      else 
        meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        const int ipNdim = ip*nDim;

        const int offSetSF = ip*nodesPerElement;

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // save off mdot
        const double tmdot = mdot[ip];

        // save off some offsets
        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        // zero out values of interest for this ip
        for ( int j = 0; j < nDim; ++j ) {
          p_uIp[j] = 0.0;
          p_coordIp[j] = 0.0;
        }

        // compute scs point values; offset to Shape Function; sneak in divU
        double muIp = 0.0;
        double divU = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function[offSetSF+ic];
          const double rAdv = p_adv_shape_function[offSetSF+ic];
          muIp += r*p_viscosity[ic];
          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_coordIp[j] += rAdv*p_coordinates[ic*nDim+j];
            const double uj = p_velocityNp1[ic*nDim+j];
            p_uIp[j] += rAdv*uj;
            divU += uj*p_dndx[offSetDnDx+j];
          }
        }

        // udotx; left and right extrapolation
        double udotx = 0.0;
        const int row_p_dudxL = il*nDim*nDim;
        const int row_p_dudxR = ir*nDim*nDim;
        for (int i = 0; i < nDim; ++i ) {
          // udotx
          const double dxi = p_coordinates[irNdim+i]-p_coordinates[ilNdim+i];
          const double ui = 0.5*(p_vrtm[ilNdim+i] + p_vrtm[irNdim+i]);
          udotx += ui*dxi;
          // extrapolation du
          p_duL[i] = 0.0;
          p_duR[i] = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxjL = p_coordIp[j] - p_coordinates[ilNdim+j];
            const double dxjR = p_coordinates[irNdim+j] - p_coordIp[j];
            p_duL[i] += dxjL*p_dudx[row_p_dudxL+i*nDim+j];
            p_duR[i] += dxjR*p_dudx[row_p_dudxR+i*nDim+j];
          }
        }

        // Peclet factor; along the edge is fine
        const double diffIp = 0.5*(p_viscosity[il]/p_densityNp1[il]
                                   + p_viscosity[ir]/p_densityNp1[ir]);
        const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
        const double om_pecfac = 1.0-pecfac;
	
        // determine limiter if applicable
        if ( useLimiter ) {
          for ( int i = 0; i < nDim; ++i ) {
            const double dq = p_velocityNp1[irNdim+i] - p_velocityNp1[ilNdim+i];
            const double dqMl = 2.0*2.0*p_duL[i] - dq;
            const double dqMr = 2.0*2.0*p_duR[i] - dq;
            p_limitL[i] = van_leer(dqMl, dq, small);
            p_limitR[i] = van_leer(dqMr, dq, small);
          }
        }
	
        // final upwind extrapolation; with limiter
        for ( int i = 0; i < nDim; ++i ) {
          p_uIpL[i] = p_velocityNp1[ilNdim+i] + p_duL[i]*hoUpwind*p_limitL[i];
          p_uIpR[i] = p_velocityNp1[irNdim+i] - p_duR[i]*hoUpwind*p_limitR[i];
        }

        // assemble advection; rhs and upwind contributions; add in divU stress (explicit)
        for ( int i = 0; i < nDim; ++i ) {

          // 2nd order central
          const double uiIp = p_uIp[i];

          // upwind
          const double uiUpwind = (tmdot > 0) ? alphaUpw*p_uIpL[i] + (om_alphaUpw)*uiIp
            : alphaUpw*p_uIpR[i] + (om_alphaUpw)*uiIp;

          // generalized central (2nd and 4th order)
          const double uiHatL = alpha*p_uIpL[i] + om_alpha*uiIp;
          const double uiHatR = alpha*p_uIpR[i] + om_alpha*uiIp;
          const double uiCds = 0.5*(uiHatL + uiHatR);

          // total advection; pressure contribution in time term
          const double aflux = tmdot*(pecfac*uiUpwind + om_pecfac*uiCds);

          // divU stress term
          const double divUstress = 2.0/3.0*muIp*divU*p_scs_areav[ipNdim+i]*includeDivU_;

          const int indexL = ilNdim + i;
          const int indexR = irNdim + i;

          const int rowL = indexL*nodesPerElement*nDim;
          const int rowR = indexR*nodesPerElement*nDim;

          const int rLiL_i = rowL+ilNdim+i;
          const int rLiR_i = rowL+irNdim+i;
          const int rRiR_i = rowR+irNdim+i;
          const int rRiL_i = rowR+ilNdim+i;

          // right hand side; L and R
          p_rhs[indexL] -= aflux + divUstress;
          p_rhs[indexR] += aflux + divUstress;

          // advection operator sens; all but central

          // upwind advection (includes 4th); left node
          const double alhsfacL = 0.5*(tmdot+std::abs(tmdot))*pecfac*alphaUpw
            + 0.5*alpha*om_pecfac*tmdot;
          p_lhs[rLiL_i] += alhsfacL;
          p_lhs[rRiL_i] -= alhsfacL;

          // upwind advection (includes 4th); right node
          const double alhsfacR = 0.5*(tmdot-std::abs(tmdot))*pecfac*alphaUpw
            + 0.5*alpha*om_pecfac*tmdot;
          p_lhs[rRiR_i] -= alhsfacR;
          p_lhs[rLiR_i] += alhsfacR;

        }

        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const int icNdim = ic*nDim;

          // advection and diffison

          // upwind (il/ir) handled above; collect terms on alpha and alphaUpw
          const double lhsfacAdv = p_adv_shape_function[offSetSF+ic]*tmdot*(pecfac*om_alphaUpw + om_pecfac*om_alpha);

          for ( int i = 0; i < nDim; ++i ) {

            const int indexL = ilNdim + i;
            const int indexR = irNdim + i;

            const int rowL = indexL*nodesPerElement*nDim;
            const int rowR = indexR*nodesPerElement*nDim;

            const int rLiC_i = rowL+icNdim+i;
            const int rRiC_i = rowR+icNdim+i;

            // advection operator  lhs; rhs handled above
            // lhs; il then ir
            p_lhs[rLiC_i] += lhsfacAdv;
            p_lhs[rRiC_i] -= lhsfacAdv;

            // viscous stress
            const int offSetDnDx = nDim*nodesPerElement*ip + icNdim;
            double lhs_riC_i = 0.0;
            for ( int j = 0; j < nDim; ++j ) {

              const double axj = p_scs_areav[ipNdim+j];
              const double uj = p_velocityNp1[icNdim+j];

              // -mu*dui/dxj*A_j; fixed i over j loop; see below..
              const double lhsfacDiff_i = -muIp*p_dndx[offSetDnDx+j]*axj;
              // lhs; il then ir
              lhs_riC_i += lhsfacDiff_i;

              // -mu*duj/dxi*A_j
              const double lhsfacDiff_j = -muIp*p_dndx[offSetDnDx+i]*axj;
              // lhs; il then ir
              p_lhs[rowL+icNdim+j] += lhsfacDiff_j;
              p_lhs[rowR+icNdim+j] -= lhsfacDiff_j;
              // rhs; il then ir
              p_rhs[indexL] -= lhsfacDiff_j*uj;
              p_rhs[indexR] += lhsfacDiff_j*uj;
            }

            // deal with accumulated lhs and flux for -mu*dui/dxj*Aj
            p_lhs[rLiC_i] += lhs_riC_i;
            p_lhs[rRiC_i] -= lhs_riC_i;
            const double ui = p_velocityNp1[icNdim+i];
            p_rhs[indexL] -= lhs_riC_i*ui;
            p_rhs[indexR] += lhs_riC_i*ui;

          }
        }
      }

      // call supplemental
      for ( size_t i = 0; i < supplementalAlgSize; ++i )
        supplementalAlg_[i]->elem_execute( &lhs[0], &rhs[0], elem, meSCS, meSCV);

      // relax the diagonal term before applying to the matrix
      const int npeNdim = nodesPerElement*nDim;
      for (int ir=0; ir < npeNdim; ir++) {
        p_lhs[ir * (npeNdim + 1)] /= relaxFacU;
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

//--------------------------------------------------------------------------
//-------- van_leer ---------------------------------------------------------
//--------------------------------------------------------------------------
double
AssembleMomentumElemSolverAlgorithm::van_leer(
  const double &dqm,
  const double &dqp,
  const double &small)
{
  double limit = (2.0*(dqm*dqp+std::abs(dqm*dqp))) /
    ((dqm+dqp)*(dqm+dqp)+small);
  return limit;
}

} // namespace nalu
} // namespace Sierra
