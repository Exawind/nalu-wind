/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <mesh_motion/AssemblePressureForceBCSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

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
// AssemblePressureForceBCSolverAlgorithm - Int bcScalarQ*area_i
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssemblePressureForceBCSolverAlgorithm::AssemblePressureForceBCSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *bcScalarQ,
  bool use_shifted_integration)
  : SolverAlgorithm(realm, part, eqSystem),
    bcScalarQ_(bcScalarQ),
    use_shifted_integration_(use_shifted_integration)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssemblePressureForceBCSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssemblePressureForceBCSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

   // space for LHS/RHS; nodesPerElem*nDim*nodesPerElem*nDim and nodesPerElem*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // nodal fields to gather
  std::vector<double> ws_face_coordinates;
  std::vector<double> ws_bcScalarQ;

  // master element
  std::vector<double> ws_face_shape_function;

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

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;


    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
    const int rhsSize = nodesPerElement*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // algorithm related; element
    ws_face_coordinates.resize(nodesPerFace*nDim);
    ws_bcScalarQ.resize(nodesPerFace);
    ws_face_shape_function.resize(nodesPerFace*nodesPerFace);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_face_coordinates = &ws_face_coordinates[0];
    double *p_bcScalarQ = &ws_bcScalarQ[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions
    if (use_shifted_integration_)
    {
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    }
    else{
      meFC->shape_fcn(&p_face_shape_function[0]);
    }

    const size_t length   = b.size();

    for ( size_t k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;

      // get face
      stk::mesh::Entity face = b[k];

      //======================================
      // gather nodal data off of face
      //======================================
      stk::mesh::Entity const * face_node_rels = bulk_data .begin_nodes(face);
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        p_bcScalarQ[ni] = *stk::mesh::field_data(*bcScalarQ_, node);

        // gather vectors
        double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_face_coordinates[offSet+i] = coords[i];
        }
      }

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
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        // set connected nodes
        connected_nodes[ni] = node;
      }

      // pointer to face data
      double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // loop over face nodes
      for ( int ip = 0; ip < num_face_nodes; ++ip ) {

        const int nearestNode = face_node_ordinals[ip];

        const int offSetSF_face = ip*nodesPerFace;

        // interpolate to bip
        double fluxBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          fluxBip += r*p_bcScalarQ[ic];
        }

        // assemble for each of the ith component
        for ( int i = 0; i < nDim; ++i ) {
          const int indexR = nearestNode*nDim + i;
          p_rhs[indexR] -= fluxBip*areaVec[ip*nDim+i];
          // RHS only, no need to populate LHS (is zeroed out)
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
    }
  }
}

} // namespace nalu
} // namespace Sierra


