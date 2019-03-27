/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <mesh_motion/AssembleMeshDisplacementElemSolverAlgorithm.h>
#include <EquationSystem.h>
#include <SolverAlgorithm.h>

#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

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
// AssembleMeshDisplacementElemSolverAlgorithm - add LHS/RHS for uvw momentum
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMeshDisplacementElemSolverAlgorithm::AssembleMeshDisplacementElemSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  const bool deformWrtModelCoords)
  : SolverAlgorithm(realm, part, eqSystem),
    deformWrtModelCoords_(deformWrtModelCoords)
{
  // save off data
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  meshDisplacement_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_displacement");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  modelCoordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // property
  mu_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "lame_mu");
  lambda_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "lame_lambda");

  /* Matrix layout follows momentum */
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMeshDisplacementElemSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMeshDisplacementElemSolverAlgorithm::execute()
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
  std::vector<double> ws_displacementNp1;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_modelCoordinates;
  std::vector<double> ws_mu;
  std::vector<double> ws_lambda;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_shape_function;

  // deal with state
  VectorFieldType &displacementNp1 = meshDisplacement_->field_of_state(stk::mesh::StateNP1);

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

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
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
    ws_displacementNp1.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_modelCoordinates.resize(nodesPerElement*nDim);
    ws_mu.resize(nodesPerElement);
    ws_lambda.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_dndx.resize(nDim*numScsIp*nodesPerElement);
    ws_deriv.resize(nDim*numScsIp*nodesPerElement);
    ws_det_j.resize(numScsIp);
    ws_shape_function.resize(numScsIp*nodesPerElement);

    // pointer to lhs/rhs
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_displacementNp1 = &ws_displacementNp1[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_modelCoordinates = &ws_modelCoordinates[0];
    double *p_mu = &ws_mu[0];
    double *p_lambda = &ws_lambda[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_dndx = &ws_dndx[0];
    double *p_shape_function = &ws_shape_function[0];

    // extract shape function
    meSCS->shape_fcn(&p_shape_function[0]);

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
        const double * dxNp1  =  stk::mesh::field_data(displacementNp1, node);
        const double * coords =  stk::mesh::field_data(*coordinates_, node);
        const double * modelCoords =  stk::mesh::field_data(*modelCoordinates_, node);
        const double mu = *stk::mesh::field_data(*mu_, node);
        const double lambda = *stk::mesh::field_data(*lambda_, node);

        // gather scalars
        p_mu[ni] = mu;
        p_lambda[ni] = lambda;

        // gather vectors
        const int niNdim = ni*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_displacementNp1[niNdim+i] = dxNp1[i];
          p_coordinates[niNdim+i] = coords[i];
          p_modelCoordinates[niNdim+i] = modelCoords[i];

        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // compute dndx; model coords or displaced?
      if ( deformWrtModelCoords_ ) {
        meSCS->grad_op(1, &p_modelCoordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      }
      else {
        meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      }
        
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        const int ipNdim = ip*nDim;

        const int offSetSF = ip*nodesPerElement;

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // save off some offsets
        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        // compute scs point values; offset to Shape Function; sneak in divU
        double muIp = 0.0;
        double lambdaIp = 0.0;
        double divDx = 0.0;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function[offSetSF+ic];
          muIp += r*p_mu[ic];
          lambdaIp += r*p_lambda[ic];
          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            const double dxj = p_displacementNp1[ic*nDim+j];
            divDx += dxj*p_dndx[offSetDnDx+j];
          }
        }

        // assemble divDx term (explicit)
        for ( int i = 0; i < nDim; ++i ) {
          // divU stress term
          const double divTerm = -lambdaIp*divDx*p_scs_areav[ipNdim+i];
          const int indexL = ilNdim + i;
          const int indexR = irNdim + i;
          // right hand side; L and R
          p_rhs[indexL] -= divTerm;
          p_rhs[indexR] += divTerm;
        }

        // stress
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const int icNdim = ic*nDim;

          for ( int i = 0; i < nDim; ++i ) {

            const int indexL = ilNdim + i;
            const int indexR = irNdim + i;

            const int rowL = indexL*nodesPerElement*nDim;
            const int rowR = indexR*nodesPerElement*nDim;

            const int rLiC_i = rowL+icNdim+i;
            const int rRiC_i = rowR+icNdim+i;

            // viscous stress
            const int offSetDnDx = nDim*nodesPerElement*ip + icNdim;
            double lhs_riC_i = 0.0;
            for ( int j = 0; j < nDim; ++j ) {

              const double axj = p_scs_areav[ipNdim+j];
              const double dxj = p_displacementNp1[icNdim+j];

              // -mu*dxi/dxj*A_j; fixed i over j loop; see below..
              const double lhsfacDiff_i = -muIp*p_dndx[offSetDnDx+j]*axj;
              // lhs; il then ir
              lhs_riC_i += lhsfacDiff_i;

              // -mu*dxj/dxi*A_j
              const double lhsfacDiff_j = -muIp*p_dndx[offSetDnDx+i]*axj;
              // lhs; il then ir
              p_lhs[rowL+icNdim+j] += lhsfacDiff_j;
              p_lhs[rowR+icNdim+j] -= lhsfacDiff_j;

              // rhs; il then ir
              p_rhs[indexL] -= lhsfacDiff_j*dxj;
              p_rhs[indexR] += lhsfacDiff_j*dxj;
            }

            // deal with accumulated lhs and flux for -mu*dxi/dxj*Aj
            p_lhs[rLiC_i] += lhs_riC_i;
            p_lhs[rRiC_i] -= lhs_riC_i;
            const double dxi = p_displacementNp1[icNdim+i];
            p_rhs[indexL] -= lhs_riC_i*dxi;
            p_rhs[indexR] += lhs_riC_i*dxi;

          }
        }
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
