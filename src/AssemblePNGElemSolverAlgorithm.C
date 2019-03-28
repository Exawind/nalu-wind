/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssemblePNGElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
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
// AssemblePNGElemSolverAlgorithm - add LHS/RHS for uvw momentum
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssemblePNGElemSolverAlgorithm::AssemblePNGElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  std::string independentDofName,
  std::string dofName)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(NULL),
    dqdx_(NULL),
    coordinates_(NULL)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  scalarQ_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, independentDofName);
  dqdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, dofName);
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssemblePNGElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssemblePNGElemSolverAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // space for LHS/RHS; nodesPerElem*nDim*nodesPerElem*nDim and nodesPerElem*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // nodal fields to gather
  std::vector<double> ws_scalarQ;
  std::vector<double> ws_dqdx;
  std::vector<double> ws_coordinates;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_scv_volume;
  std::vector<double> ws_shape_function_scs;
  std::vector<double> ws_shape_function_scv;

  // fixed size with pointers
  std::vector<double> dqdxScv(nDim,0.0);
  double *p_dqdxScv = &dqdxScv[0];

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

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
    const int numScvIp = meSCV->num_integration_points();

    // mappings for this element, SCS and SCV
    const int *lrscv = meSCS->adjacentNodes();
    const int *ipNodeMap = meSCV->ipNodeMap();

    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
    const int rhsSize = nodesPerElement*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // algorithm related
    ws_scalarQ.resize(nodesPerElement);
    ws_dqdx.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_scv_volume.resize(numScvIp);
    ws_shape_function_scs.resize(numScsIp*nodesPerElement);
    ws_shape_function_scv.resize(numScvIp*nodesPerElement);

    // pointer to lhs/rhs
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_scalarQ = &ws_scalarQ[0];
    double *p_dqdx = &ws_dqdx[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_scv_volume = &ws_scv_volume[0];
    double *p_shape_function_scs = &ws_shape_function_scs[0];
    double *p_shape_function_scv = &ws_shape_function_scv[0];

    // extract shape function
    meSCS->shape_fcn(&p_shape_function_scs[0]);
    meSCV->shape_fcn(&p_shape_function_scv[0]);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // zero lhs/rhs
      for ( int p = 0; p < lhsSize; ++p )
        p_lhs[p] = 0.0;
      for ( int p = 0; p < rhsSize; ++p )
        p_rhs[p] = 0.0;
      
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
        const double scalarQ   = *stk::mesh::field_data(*scalarQ_, node);
        const double * dqdx   =  stk::mesh::field_data(*dqdx_, node);
        const double * coords =  stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        p_scalarQ[ni] = scalarQ;

        // gather vectors
        const int niNdim = ni*nDim;

        for ( int i=0; i < nDim; ++i ) {
          p_dqdx[niNdim+i] = dqdx[i];
          p_coordinates[niNdim+i] = coords[i];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      double scv_error = 0.0;
      meSCV->determinant(1, &p_coordinates[0], &p_scv_volume[0], &scv_error);

      // handle scs first; all RHS as if it is a source term
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        const int ipNdim = ip*nDim;

        const int offSetSF = ip*nodesPerElement;

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // save off some offsets
        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        // compute scs point values
        double scalarQIp = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function_scs[offSetSF+ic];
          scalarQIp += r*p_scalarQ[ic];
        }
      
        // add residual for each component i
        for ( int i = 0; i < nDim; ++i ) {  
          const int indexL = ilNdim + i;
          const int indexR = irNdim + i;

          const double axi = p_scs_areav[ipNdim+i];

          // right hand side; L and R
          const double rhsFac = -scalarQIp*axi;
          p_rhs[indexL] -= rhsFac;
          p_rhs[indexR] += rhsFac;  
        }
      }

      // handle scv LHS second
      for ( int ip = 0; ip < numScvIp; ++ip ) {

        // nearest node to ip
        const int nearestNode = ipNodeMap[ip];

        // save off some offsets and sc_volume at this ip
        const int nnNdim = nearestNode*nDim;
        const int offSetSF = ip*nodesPerElement;
        const double scV = p_scv_volume[ip];

        // zero out scv
        for ( int j = 0; j < nDim; ++j )
          p_dqdxScv[j] = 0.0;
        
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function_scv[offSetSF+ic];
          for ( int j = 0; j < nDim; ++j ) {
            p_dqdxScv[j] += r*p_dqdx[ic*nDim+j];
          }
        }
      
        // assemble rhs
        for ( int i = 0; i < nDim; ++i ) {
          p_rhs[nnNdim+i] -= p_dqdxScv[i]*scV;
        }

        // manage LHS
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          
          const int icNdim = ic*nDim;
      
          // save off shape function
          const double r = p_shape_function_scv[offSetSF+ic];
      
          const double lhsfac = r*scV;
      
          for ( int i = 0; i < nDim; ++i ) {
            const int indexNN = nnNdim + i;
            const int rowNN = indexNN*nodesPerElement*nDim;
            const int rNNiC_i = rowNN+icNdim+i;
            p_lhs[rNNiC_i] += lhsfac;
          }
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
