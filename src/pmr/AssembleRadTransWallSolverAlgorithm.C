/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <pmr/AssembleRadTransWallSolverAlgorithm.h>
#include <pmr/RadiativeTransportEquationSystem.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleRadTransWallSolverAlgorithm - weak form for intensity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleRadTransWallSolverAlgorithm::AssembleRadTransWallSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  RadiativeTransportEquationSystem *radEqSystem,
  const bool &useShifted)
  : SolverAlgorithm(realm, part, radEqSystem),
    radEqSystem_(radEqSystem),
    useShifted_(useShifted),
    intensity_(NULL),
    bcIntensity_(NULL),
    exposedAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  bcIntensity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity_bc");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransWallSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleRadTransWallSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract current ordinate direction
  std::vector<double> Sk(nDim,0.0);
  radEqSystem_->get_current_ordinate(&Sk[0]);
  const double *p_Sk = &Sk[0];
  intensity_ = radEqSystem_->get_intensity();

  // space for LHS/RHS; nodesPerFace*nodesPerFace and nodesPerFace
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // nodal fields to gather
  std::vector<double> ws_intensity;
  std::vector<double> ws_bcIntensity;

  // master element
  std::vector<double> ws_face_shape_function;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;

    // resize some things; matrix related
    const int lhsSize = nodesPerFace*nodesPerFace;
    const int rhsSize = nodesPerFace;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerFace);

    // algorithm related; element
    ws_intensity.resize(nodesPerFace);
    ws_bcIntensity.resize(nodesPerFace);
    ws_face_shape_function.resize(nodesPerFace*nodesPerFace);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_intensity = &ws_intensity[0];
    double *p_bcIntensity = &ws_bcIntensity[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions
    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

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
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);
      for ( int ni = 0; ni < nodesPerFace; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        connected_nodes[ni] = node;
        // gather scalars
        p_intensity[ni]    = *stk::mesh::field_data(*intensity_, node);
        p_bcIntensity[ni]  = *stk::mesh::field_data(*bcIntensity_, node);
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // loop over face nodes
      for ( int ip = 0; ip < nodesPerFace; ++ip ) {

        // nearest node (to which we will assemble RHS); offsets
        const int nearestNode = ip;
        const int offSetAveraVec = ip*nDim;
        const int offSetSF_face = ip*nodesPerFace;

        // interpolate to bip; both intensity flavors
        double iBip = 0.0;
        double iBcBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          iBip += r*p_intensity[ic];
          iBcBip += r*p_bcIntensity[ic];
        }

        // determine in or out intensity bc based on sign of ajsj
        double ajsj = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          ajsj += areaVec[offSetAveraVec+j]*p_Sk[j];
        }

        // we are simply forming Int sj I njdS; flavor of I depends on sign of ajsj
        if ( ajsj > 0 ) {
          // use dof intensity; requires LHS assemble
          int rowR = nearestNode*nodesPerFace;
          for (int ic = 0; ic < nodesPerFace; ++ic)
            p_lhs[rowR+ic] += ajsj*p_face_shape_function[offSetSF_face+ic];

          p_rhs[nearestNode] -= iBip*ajsj;

        }
        else {
          // use bc intensity; requires NO LHS assembly
          p_rhs[nearestNode] -= iBcBip*ajsj;
        }

      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}


} // namespace nalu
} // namespace Sierra
