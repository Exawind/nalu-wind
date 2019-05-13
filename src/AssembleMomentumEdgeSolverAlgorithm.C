/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumEdgeSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumEdgeSolverAlgorithm - add LHS/RHS for momentum
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumEdgeSolverAlgorithm::AssembleMomentumEdgeSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm_.does_mesh_move()),
    includeDivU_(realm_.get_divU()),
    velocityRTM_(NULL),
    velocity_(NULL),
    coordinates_(NULL),
    dudx_(NULL),
    density_(NULL),
    viscosity_(NULL),
    edgeAreaVec_(NULL),
    massFlowRate_(NULL),
    pecletFunction_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  // extract viscosity  name
  const std::string viscName = realm_.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
  massFlowRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function<double>(velocity_->name());
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumEdgeSolverAlgorithm::~AssembleMomentumEdgeSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = "velocity";
  const double alpha = realm_.get_alpha_factor(dofName);
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const double relaxFacU = realm_.solutionOptions_->get_relaxation_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);

  // one minus flavor
  const double om_alpha = 1.0-alpha;
  const double om_alphaUpw = 1.0-alphaUpw;

  // space for LHS/RHS; always edge connectivity
  const int nodesPerEdge = 2;
  const int lhsSize = nDim*nodesPerEdge*nDim*nodesPerEdge;
  const int rhsSize = nDim*nodesPerEdge;
  std::vector<double> lhs(lhsSize);
  std::vector<double> rhs(rhsSize);
  std::vector<int> scratchIds(rhsSize);
  std::vector<double> scratchVals(rhsSize);
  std::vector<stk::mesh::Entity> connected_nodes(2);

  // area vector; gather into
  std::vector<double> areaVec(nDim);

  // pointer for fast access
  double *p_lhs = &lhs[0];
  double *p_rhs = &rhs[0];
  double *p_areaVec = &areaVec[0];

  // space for dui/dxj. This variable is the modified gradient with NOC
  std::vector<double> duidxj(nDim*nDim);

  // extrapolated value from the L/R direction 
  std::vector<double> uIpL(nDim);
  std::vector<double> uIpR(nDim);
  // limiter values from the L/R direction, 0:1
  std::vector<double> limitL(nDim,1.0); 
  std::vector<double> limitR(nDim,1.0);
  // extrapolated gradient from L/R direction
  std::vector<double> duL(nDim);
  std::vector<double> duR(nDim);
  
  // pointers for fast access
  double *p_duidxj = &duidxj[0];
  double *p_uIpL = &uIpL[0];
  double *p_uIpR = &uIpR[0];
  double *p_limitL = &limitL[0];
  double *p_limitR = &limitR[0];
  double *p_duL = &duL[0];
  double *p_duR = &duR[0];

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // pointer to edge area vector and mdot
    const double * av = stk::mesh::field_data(*edgeAreaVec_, b);
    const double * mdot = stk::mesh::field_data(*massFlowRate_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zeroing of lhs/rhs
      for ( int i = 0; i < lhsSize; ++i ) {
        p_lhs[i] = 0.0;
      }
      for ( int i = 0; i < rhsSize; ++i ) {
        p_rhs[i] = 0.0;
      }

      stk::mesh::Entity const * edge_node_rels = b.begin_nodes(k);

      // pointer to edge area vector
      for ( int j = 0; j < nDim; ++j )
        p_areaVec[j] = av[k*nDim+j];
      const double tmdot = mdot[k];

      // sanity check on number or nodes
      ThrowAssert( b.num_nodes(k) == 2 );

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      const double * coordL = stk::mesh::field_data(*coordinates_, nodeL);
      const double * coordR = stk::mesh::field_data(*coordinates_, nodeR);

      const double * dudxL = stk::mesh::field_data(*dudx_, nodeL);
      const double * dudxR = stk::mesh::field_data(*dudx_, nodeR);

      const double * vrtmL = stk::mesh::field_data(*velocityRTM_, nodeL);
      const double * vrtmR = stk::mesh::field_data(*velocityRTM_, nodeR);

      const double * uNp1L = stk::mesh::field_data(velocityNp1, nodeL);
      const double * uNp1R = stk::mesh::field_data(velocityNp1, nodeR);

      const double densityL = *stk::mesh::field_data(densityNp1, nodeL);
      const double densityR = *stk::mesh::field_data(densityNp1, nodeR);

      const double viscosityL = *stk::mesh::field_data(*viscosity_, nodeL);
      const double viscosityR = *stk::mesh::field_data(*viscosity_, nodeR);

      // copy in extrapolated values
      for ( int i = 0; i < nDim; ++i ) {
        // extrapolated du
        p_duL[i] = 0.0;
        p_duR[i] = 0.0;
        const int offSet = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = 0.5*(coordR[j] - coordL[j]);
          p_duL[i] += dxj*dudxL[offSet+j];
          p_duR[i] += dxj*dudxR[offSet+j];
        }
      }

      // compute geometry
      double axdx = 0.0;
      double asq = 0.0;
      double udotx = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        axdx += axj*dxj;
        asq += axj*axj;
        udotx += 0.5*dxj*(vrtmL[j] + vrtmR[j]);
      }

      const double inv_axdx = 1.0/axdx;

      // ip props
      const double viscIp = 0.5*(viscosityL + viscosityR);
      const double diffIp = 0.5*(viscosityL/densityL + viscosityR/densityR);

      // Peclet factor
      const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
      const double om_pecfac = 1.0-pecfac;

      // determine limiter if applicable
      if ( useLimiter ) {
        for ( int i = 0; i < nDim; ++i ) {
          const double dq = uNp1R[i] - uNp1L[i];
          const double dqMl = 2.0*2.0*p_duL[i] - dq;
          const double dqMr = 2.0*2.0*p_duR[i] - dq;
          p_limitL[i] = van_leer(dqMl, dq, small);
          p_limitR[i] = van_leer(dqMr, dq, small);
        }
      }

      // final upwind extrapolation; with limiter
      for ( int i = 0; i < nDim; ++i ) {
        p_uIpL[i] = uNp1L[i] + p_duL[i]*hoUpwind*p_limitL[i];
        p_uIpR[i] = uNp1R[i] - p_duR[i]*hoUpwind*p_limitR[i];
      }

      /*
        form duidxj with over-relaxed procedure of Jasak:

        dui/dxj = GjUi +[(uiR - uiL) - GlUi*dxl]*Aj/AxDx
        where Gp is the interpolated pth nodal gradient for ui
      */
      for ( int i = 0; i < nDim; ++i ) {

        // difference between R and L nodes for component i
        const double uidiff = uNp1R[i] - uNp1L[i];

        // offset into all forms of dudx
        const int offSetI = nDim*i;

        // start sum for NOC contribution
        double GlUidxl = 0.0;
        for ( int l = 0; l< nDim; ++l ) {
          const int offSetIL = offSetI+l;
          const double dxl = coordR[l] - coordL[l];
          const double GlUi = 0.5*(dudxL[offSetIL] + dudxR[offSetIL]);
          GlUidxl += GlUi*dxl;
        }

        // form full tensor dui/dxj with NOC
        for ( int j = 0; j < nDim; ++j ) {
          const int offSetIJ = offSetI+j;
          const double axj = p_areaVec[j];
          const double GjUi = 0.5*(dudxL[offSetIJ] + dudxR[offSetIJ]);
          p_duidxj[offSetIJ] = GjUi + (uidiff - GlUidxl)*axj*inv_axdx;
        }
      }

      // lhs diffusion; only -mu*dui/dxj*Aj contribution for now
      const double dlhsfac = -viscIp*asq*inv_axdx;

      for ( int i = 0; i < nDim; ++i ) {

        // 2nd order central
        const double uiIp = 0.5*(uNp1R[i] + uNp1L[i]);

        // upwind
        const double uiUpwind = (tmdot > 0) ? alphaUpw*p_uIpL[i] + om_alphaUpw*uiIp
          : alphaUpw*p_uIpR[i] + om_alphaUpw*uiIp;

        // generalized central (2nd and 4th order)
        const double uiHatL = alpha*p_uIpL[i] + om_alpha*uiIp;
        const double uiHatR = alpha*p_uIpR[i] + om_alpha*uiIp;
        const double uiCds = 0.5*(uiHatL + uiHatR);

        // total advection; pressure contribution in time term expression
        const double aflux = tmdot*(pecfac*uiUpwind + om_pecfac*uiCds);

        // divU
        double divU = 0.0;
        for ( int j = 0; j < nDim; ++j)
          divU += p_duidxj[j*nDim+j];

        // diffusive flux; viscous tensor doted with area vector
        double dflux = 2.0/3.0*viscIp*divU*p_areaVec[i]*includeDivU_;
        const int offSetI = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const int offSetTrans = nDim*j+i;
          const double axj = p_areaVec[j];
          dflux += -viscIp*(p_duidxj[offSetI+j] + p_duidxj[offSetTrans])*axj;
        }

        // residal for total flux
        const double tflux = aflux + dflux;
        const int indexL = i;
        const int indexR = i + nDim;

        // total flux left
        p_rhs[indexL] -= tflux;
        // total flux right
        p_rhs[indexR] += tflux;

        // setup for LHS
        const int rowL = indexL * nodesPerEdge*nDim;
        const int rowR = indexR * nodesPerEdge*nDim;

        //==============================
        // advection first
        //==============================
        const int rLiL = rowL+indexL;
        const int rLiR = rowL+indexR;
        const int rRiL = rowR+indexL;
        const int rRiR = rowR+indexR;

        // upwind advection (includes 4th); left node
        double alhsfac = 0.5*(tmdot+std::abs(tmdot))*pecfac*alphaUpw
          + 0.5*alpha*om_pecfac*tmdot;
        p_lhs[rLiL] += alhsfac / relaxFacU;
        p_lhs[rRiL] -= alhsfac;

        // upwind advection (incldues 4th); right node
        alhsfac = 0.5*(tmdot-std::abs(tmdot))*pecfac*alphaUpw
          + 0.5*alpha*om_pecfac*tmdot;
        p_lhs[rRiR] -= alhsfac / relaxFacU;
        p_lhs[rLiR] += alhsfac;

        // central; left; collect terms on alpha and alphaUpw
        alhsfac = 0.5*tmdot*(pecfac*om_alphaUpw + om_pecfac*om_alpha);
        p_lhs[rLiL] += alhsfac / relaxFacU;
        p_lhs[rLiR] += alhsfac;
        // central; right
        p_lhs[rRiL] -= alhsfac;
        p_lhs[rRiR] -= alhsfac / relaxFacU;

        //==============================
        // diffusion second
        //==============================
        const double axi = p_areaVec[i];

        //diffusion; row IL
        p_lhs[rLiL] -= dlhsfac / relaxFacU;
        p_lhs[rLiR] += dlhsfac;

        // diffusion; row IR
        p_lhs[rRiL] += dlhsfac;
        p_lhs[rRiR] -= dlhsfac / relaxFacU;

        // more diffusion; see theory manual
        for ( int j = 0; j < nDim; ++j ) {
          const double lhsfacNS = -viscIp*axi*p_areaVec[j]*inv_axdx;

          const int colL = j;
          const int colR = j + nDim;

          // first left; IL,IL; IL,IR
          p_lhs[rowL + colL] -= lhsfacNS / relaxFacU;
          p_lhs[rowL + colR] += lhsfacNS;

          // now right, IR,IL; IR,IR
          p_lhs[rowR + colL] += lhsfacNS;
          p_lhs[rowR + colR] -= lhsfacNS / relaxFacU;
        }

      }
      
      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

//--------------------------------------------------------------------------
//-------- van_leer ---------------------------------------------------------
//--------------------------------------------------------------------------
double
AssembleMomentumEdgeSolverAlgorithm::van_leer(
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
