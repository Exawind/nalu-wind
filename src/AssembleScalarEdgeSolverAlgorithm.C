/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarEdgeSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>

// stk_mesh/base/fem
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
// AssembleScalarEdgeSolverAlgorithm - add LHS/RHS for scalar
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarEdgeSolverAlgorithm::AssembleScalarEdgeSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm_.does_mesh_move()),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    diffFluxCoeff_(diffFluxCoeff),
    velocityRTM_(NULL),
    coordinates_(NULL),
    density_(NULL),
    massFlowRate_(NULL),
    edgeAreaVec_(NULL),
    pecletFunction_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  massFlowRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate");
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function<double>(scalarQ_->name());
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarEdgeSolverAlgorithm::~AssembleScalarEdgeSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarEdgeSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = scalarQ_->name();
  const double alpha = realm_.get_alpha_factor(dofName);
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const double relaxFac = realm_.solutionOptions_->get_relaxation_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);

  // one minus flavor
  const double om_alpha = 1.0-alpha;
  const double om_alphaUpw = 1.0-alphaUpw;

  // space for LHS/RHS; always edge connectivity
  const int nodesPerEdge = 2;
  const int lhsSize = nodesPerEdge*nodesPerEdge;
  const int rhsSize = nodesPerEdge;
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

  // deal with state
  ScalarFieldType &scalarQNp1  = scalarQ_->field_of_state(stk::mesh::StateNP1);
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

      // get edge
      stk::mesh::Entity edge = b[k];

      stk::mesh::Entity const * edge_node_rels = bulk_data.begin_nodes(edge);

      // sanity check on number or nodes
      ThrowAssert( bulk_data.num_nodes(edge) == 2 );

      // pointer to edge area vector
      for ( int j = 0; j < nDim; ++j )
        p_areaVec[j] = av[k*nDim+j];
      const double tmdot = mdot[k];

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      const double * coordL = stk::mesh::field_data(*coordinates_, nodeL);
      const double * coordR = stk::mesh::field_data(*coordinates_, nodeR);

      const double * dqdxL = stk::mesh::field_data(*dqdx_, nodeL);
      const double * dqdxR = stk::mesh::field_data(*dqdx_, nodeR);

      const double * vrtmL = stk::mesh::field_data(*velocityRTM_, nodeL);
      const double * vrtmR = stk::mesh::field_data(*velocityRTM_, nodeR);

      const double qNp1L = *stk::mesh::field_data(scalarQNp1, nodeL);
      const double qNp1R = *stk::mesh::field_data(scalarQNp1, nodeR);

      const double densityL = *stk::mesh::field_data(densityNp1, nodeL);
      const double densityR = *stk::mesh::field_data(densityNp1, nodeR);

      const double diffFluxCoeffL = *stk::mesh::field_data(*diffFluxCoeff_, nodeL);
      const double diffFluxCoeffR = *stk::mesh::field_data(*diffFluxCoeff_, nodeR);

      // compute geometry
      double axdx = 0.0;
      double asq = 0.0;
      double udotx = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        asq += axj*axj;
        axdx += axj*dxj;
        udotx += 0.5*dxj*(vrtmL[j] + vrtmR[j]);
      }

      const double inv_axdx = 1.0/axdx;

      // ip props
      const double viscIp = 0.5*(diffFluxCoeffL + diffFluxCoeffR);
      const double diffIp = 0.5*(diffFluxCoeffL/densityL + diffFluxCoeffR/densityR);

      // Peclet factor
      const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
      const double om_pecfac = 1.0-pecfac;

      // left and right extrapolation; add in diffusion calc
      double dqL = 0.0;
      double dqR = 0.0;
      double nonOrth = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double dxj = coordR[j] - coordL[j];
        dqL += 0.5*dxj*dqdxL[j];
        dqR += 0.5*dxj*dqdxR[j];
        // now non-orth (over-relaxed procedure of Jasek)
        const double axj = p_areaVec[j];
        const double kxj = axj - asq*inv_axdx*dxj;
        const double GjIp = 0.5*(dqdxL[j] + dqdxR[j]);
        nonOrth += -viscIp*kxj*GjIp;
      }

      // add limiter if appropriate
      double limitL = 1.0;
      double limitR = 1.0;
      const double dq = qNp1R - qNp1L;
      if ( useLimiter ) {
        const double dqMl = 2.0*2.0*dqL - dq;
        const double dqMr = 2.0*2.0*dqR - dq;
        limitL = van_leer(dqMl, dq, small);
        limitR = van_leer(dqMr, dq, small);
      }
      
      // extrapolated; for now limit
      const double qIpL = qNp1L + dqL*hoUpwind*limitL;
      const double qIpR = qNp1R - dqR*hoUpwind*limitR;

      //====================================
      // diffusive flux
      //====================================
      double lhsfac = -viscIp*asq*inv_axdx;
      double diffFlux = lhsfac*(qNp1R - qNp1L) + nonOrth;

      // first left
      p_lhs[0] = -lhsfac / relaxFac;
      p_lhs[1] = +lhsfac;
      p_rhs[0] = -diffFlux;

      // now right
      p_lhs[2] = +lhsfac;
      p_lhs[3] = -lhsfac / relaxFac;
      p_rhs[1] = diffFlux;

      //====================================
      // advective flux
      //====================================

      // 2nd order central
      const double qIp = 0.5*( qNp1L + qNp1R );

      // upwind
      const double qUpwind = (tmdot > 0) ? alphaUpw*qIpL + om_alphaUpw*qIp
          : alphaUpw*qIpR + om_alphaUpw*qIp;

      // generalized central (2nd and 4th order)
      const double qHatL = alpha*qIpL + om_alpha*qIp;
      const double qHatR = alpha*qIpR + om_alpha*qIp;
      const double qCds = 0.5*(qHatL + qHatR);

      // total advection
      const double aflux = tmdot*(pecfac*qUpwind + om_pecfac*qCds);

      // upwind advection (includes 4th); left node
      double alhsfac = 0.5*(tmdot+std::abs(tmdot))*pecfac*alphaUpw
        + 0.5*alpha*om_pecfac*tmdot;
      p_lhs[0] += alhsfac / relaxFac;
      p_lhs[2] -= alhsfac;

      // upwind advection; right node
      alhsfac = 0.5*(tmdot-std::abs(tmdot))*pecfac*alphaUpw
        + 0.5*alpha*om_pecfac*tmdot;
      p_lhs[3] -= alhsfac / relaxFac;
      p_lhs[1] += alhsfac;

      // central; left; collect terms on alpha and alphaUpw
      alhsfac = 0.5*tmdot*(pecfac*om_alphaUpw + om_pecfac*om_alpha);
      p_lhs[0] += alhsfac / relaxFac;
      p_lhs[1] += alhsfac;
      // central; right; collect terms on alpha and alphaUpw
      p_lhs[2] -= alhsfac;
      p_lhs[3] -= alhsfac / relaxFac;

      // total flux left
      p_rhs[0] -= aflux;
      // total flux right
      p_rhs[1] += aflux;

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

//--------------------------------------------------------------------------
//-------- van_leer ---------------------------------------------------------
//--------------------------------------------------------------------------
double
AssembleScalarEdgeSolverAlgorithm::van_leer(
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
