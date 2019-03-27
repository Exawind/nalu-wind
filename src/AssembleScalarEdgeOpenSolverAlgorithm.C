/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarEdgeOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

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
// AssembleScalarEdgeOpenSolverAlgorithm - lhs for scalar open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarEdgeOpenSolverAlgorithm::AssembleScalarEdgeOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  ScalarFieldType *bcScalarQ,
  VectorFieldType *dqdx,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(scalarQ),
    bcScalarQ_(bcScalarQ),
    dqdx_(dqdx),
    diffFluxCoeff_(diffFluxCoeff),
    coordinates_(NULL),
    openMassFlowRate_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarEdgeOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarEdgeOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // space for LHS/RHS; nodesPerElement*nodesPerElement and nodesPerElement
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // deal with state
  ScalarFieldType &scalarQNp1 = scalarQ_->field_of_state(stk::mesh::StateNP1);
  const std::string dofName = scalarQ_->name();
  const double relaxFac = realm_.solutionOptions_->get_relaxation_factor(dofName);

  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract connected element topology
    b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
    ThrowAssert ( parentTopo.size() == 1 );
    stk::topology theElemTopo = parentTopo[0];

    // volume master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);
    const int nodesPerElement = meSCS->nodesPerElement_;


    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nodesPerElement;
    const int rhsSize = nodesPerElement;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];

    // size some things that are useful
    const int num_face_nodes = b.topology().num_nodes();
    

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // get face
      stk::mesh::Entity face = b[k];

      // pointer to face data
      const double * mdot = stk::mesh::field_data(*openMassFlowRate_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      //==========================================
      // gather nodal data off of element; n/a
      //==========================================
      const stk::mesh::Entity* elem_node_rels = bulk_data.begin_nodes(element);
      const int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        // set connected nodes
        connected_nodes[ni] = node;
      }

      // loop over face nodes
      for ( int ip = 0; ip < num_face_nodes; ++ip ) {

        const int nearestNode = face_node_ordinals[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        const double qR = *stk::mesh::field_data( scalarQNp1, nodeR );
        const double qEntrain = *stk::mesh::field_data( *bcScalarQ_, nodeR );

        //================================
        // advection first (and only)
        //================================
        const double tmdot = mdot[ip];

        const int rowR = nearestNode*nodesPerElement;

        // advection; leaving the domain
        if ( tmdot > 0.0 ) {

          // total advection
          const double aflux = tmdot*qR;

          p_rhs[nearestNode] -= aflux;

          // upwind lhs
          p_lhs[rowR+nearestNode] += tmdot / relaxFac;

        }
        else {
          // extrainment; advect in from specified value
          const double aflux = tmdot*qEntrain;
          p_rhs[nearestNode] -= aflux;
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
    }
  }
}

} // namespace nalu
} // namespace Sierra
