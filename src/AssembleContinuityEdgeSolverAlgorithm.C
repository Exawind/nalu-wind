/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleContinuityEdgeSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>

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
// AssembleContinuityEdgeSolverAlgorithm - add LHS/RHS for continuity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleContinuityEdgeSolverAlgorithm::AssembleContinuityEdgeSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    edgeAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityEdgeSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract noc
  const std::string dofName = "pressure";
  const double nocFac
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // Classic Nalu projection timescale
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double tauScale = dt / gamma1;

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;
  
  // space for LHS/RHS; always nodesPerEdge*nodesPerEdge and nodesPerEdge
  std::vector<double> lhs(4);
  std::vector<double> rhs(2);
  std::vector<int> scratchIds(2);
  std::vector<double> scratchVals(2);
  std::vector<stk::mesh::Entity> connected_nodes(2);

  // area vector; gather into
  std::vector<double> areaVec(nDim);

  // pointers for fast access
  double *p_lhs = &lhs[0];
  double *p_rhs = &rhs[0];
  double *p_areaVec = &areaVec[0];

  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType* Udiag = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "momentum_diag");

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

    // pointer to edge area vector
    const double * av = stk::mesh::field_data(*edgeAreaVec_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // sanity check on number or nodes
      ThrowAssert( b.num_nodes(k) == 2 );

      stk::mesh::Entity const * edge_node_rels = b.begin_nodes(k);

      // pointer to edge area vector
      for ( int j = 0; j < nDim; ++j )
        p_areaVec[j] = av[k*nDim+j];

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      const double * coordL = stk::mesh::field_data(*coordinates_, nodeL);
      const double * coordR = stk::mesh::field_data(*coordinates_, nodeR);

      const double * GpdxL = stk::mesh::field_data(*Gpdx_, nodeL);
      const double * GpdxR = stk::mesh::field_data(*Gpdx_, nodeR);

      const double * vrtmL = stk::mesh::field_data(*velocityRTM_, nodeL);
      const double * vrtmR = stk::mesh::field_data(*velocityRTM_, nodeR);

      const double pressureL = *stk::mesh::field_data(*pressure_, nodeL);
      const double pressureR = *stk::mesh::field_data(*pressure_, nodeR);

      const double densityL = *stk::mesh::field_data(densityNp1, nodeL);
      const double densityR = *stk::mesh::field_data(densityNp1, nodeR);

      const double udiagL = *stk::mesh::field_data(*Udiag, nodeL);
      const double udiagR = *stk::mesh::field_data(*Udiag, nodeR);

      const double projTimeScale = 0.5 * (1.0/udiagL + 1.0/udiagR);

      // compute geometry
      double axdx = 0.0;
      double asq = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        asq += axj*axj;
        axdx += axj*dxj;
      }

      const double inv_axdx = 1.0/axdx;
      const double rhoIp = 0.5*(densityR + densityL);

      //  mdot
      double tmdot = -projTimeScale*(pressureR - pressureL)*asq*inv_axdx;
      for ( int j = 0; j < nDim; ++j ) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        const double kxj = axj - asq*inv_axdx*dxj; // NOC
        const double rhoUjIp = 0.5*(densityR*vrtmR[j] + densityL*vrtmL[j]);
        const double ujIp = 0.5*(vrtmR[j] + vrtmL[j]);
        const double GjIp = 0.5*(GpdxR[j] / udiagR + GpdxL[j] / udiagL);
        tmdot += (interpTogether*rhoUjIp + om_interpTogether*rhoIp*ujIp + GjIp)*axj
          - kxj*GjIp*nocFac;
      }

      tmdot /= tauScale;
      const double lhsfac = -asq*inv_axdx*projTimeScale / tauScale;

      /*
        lhs[0] = IL,IL; lhs[1] = IL,IR; IR,IL; IR,IR
      */

      // first left
      p_lhs[0] = -lhsfac;
      p_lhs[1] = +lhsfac;
      p_rhs[0] = -tmdot;

      // now right
      p_lhs[2] = +lhsfac;
      p_lhs[3] = -lhsfac;
      p_rhs[1] = tmdot;

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
