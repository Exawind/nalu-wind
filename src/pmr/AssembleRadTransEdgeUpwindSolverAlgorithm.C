// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



// nalu
#include <pmr/AssembleRadTransEdgeUpwindSolverAlgorithm.h>
#include <pmr/RadiativeTransportEquationSystem.h>
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
// AssembleRadTransEdgeUpwindSolverAlgorithm - add LHS/RHS for continuity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleRadTransEdgeUpwindSolverAlgorithm::AssembleRadTransEdgeUpwindSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  RadiativeTransportEquationSystem *radEqSystem)
  : SolverAlgorithm(realm, part, radEqSystem),
    radEqSystem_(radEqSystem),
    intensity_(NULL),
    edgeAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  edgeAreaVec_ = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransEdgeUpwindSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransEdgeUpwindSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract current ordinate direction
  std::vector<double> Sk(nDim,0.0);
  radEqSystem_->get_current_ordinate(&Sk[0]);
  const double *p_Sk = &Sk[0];
  intensity_ = radEqSystem_->get_intensity();

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

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const size_t length   = b.size();

    // pointer to edge area vector
    const double * av = stk::mesh::field_data(*edgeAreaVec_, b);

    for ( size_t k = 0 ; k < length ; ++k ) {

      // set ordinal for edge
      unsigned edge_ordinal = k;
      // sanity check on number or nodes
      ThrowAssert( b.num_nodes(edge_ordinal) == 2 );

      stk::mesh::Entity const * edge_node_rels = b.begin_nodes(edge_ordinal);

      // pointer to edge area vector
      for ( int j = 0; j < nDim; ++j )
        p_areaVec[j] = av[k*nDim+j];

      // left and right nodes
      stk::mesh::Entity nodeL = edge_node_rels[0];
      stk::mesh::Entity nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      // extract nodal fields
      const double intensityL = *stk::mesh::field_data(*intensity_, nodeL);
      const double intensityR = *stk::mesh::field_data(*intensity_, nodeR);

      // compute sj*njdS
      double sjaj = 0.0;
      for ( int j = 0; j < nDim; ++j ) {
        sjaj += p_Sk[j]*p_areaVec[j];
      }

      // upwind; left node
      double lhsfac = 0.5*(sjaj+std::abs(sjaj));
      p_lhs[0] = +lhsfac;
      p_lhs[2] = -lhsfac;

      // upwind; right node
      lhsfac = 0.5*(sjaj-std::abs(sjaj));
      p_lhs[3] = -lhsfac;
      p_lhs[1] = +lhsfac;

      // residual
      const double intensityIp = (sjaj > 0.0) ? intensityL : intensityR;
      p_rhs[0] = -sjaj*intensityIp;
      p_rhs[1] = +sjaj*intensityIp;
      
      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
