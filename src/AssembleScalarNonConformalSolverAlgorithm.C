/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleScalarNonConformalSolverAlgorithm.h>
#include <EquationSystem.h>
#include <DgInfo.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <NonConformalInfo.h>
#include <NonConformalManager.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleScalarNonConformalSolverAlgorithm - lhs for NC bc (DG)
//                                                     used for both edge
//                                                     and element; WIP..
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleScalarNonConformalSolverAlgorithm::AssembleScalarNonConformalSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff)
  : SolverAlgorithm(realm, part, eqSystem),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff),
    coordinates_(NULL),
    exposedAreaVec_(NULL),
    ncMassFlowRate_(NULL),
    eta_(realm_.get_nc_alg_upwind_advection() ? 1.0 : 0.0),
    useCurrentNormal_(realm_.get_nc_alg_current_normal())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");  
  ncMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "nc_mass_flow_rate");

  // what do we need ghosted for this alg to work?
  ghostFieldVec_.push_back(&(scalarQ_->field_of_state(stk::mesh::StateNP1)));
  ghostFieldVec_.push_back(diffFluxCoeff_);
  ghostFieldVec_.push_back(coordinates_);
 
  // provide output to user
  if ( useCurrentNormal_ ) 
    NaluEnv::self().naluOutputP0() << "AssembleScalarNonConformalSolverAlgorithm::Options: use_current_normal is active" << std::endl;   
  if ( realm_.get_nc_alg_upwind_advection() ) 
    NaluEnv::self().naluOutputP0() << "AssembleScalarNonConformalSolverAlgorithm::Options: upwind advective flux is active " << std::endl;   
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarNonConformalSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildNonConformalNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleScalarNonConformalSolverAlgorithm::execute()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const std::string dofName = scalarQ_->name();
  const double relaxFac = realm_.solutionOptions_->get_relaxation_factor(dofName);

  // space for LHS/RHS; nodesPerElem*nodesPerElem and nodesPerElem
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;
 
  // ip values; both boundary and opposing surface
  std::vector<double> currentIsoParCoords(nDim);
  std::vector<double> opposingIsoParCoords(nDim);
  std::vector<double> cNx(nDim);
  std::vector<double> oNx(nDim);

  // mapping for -1:1 -> -0.5:0.5 volume element
  std::vector<double> currentElementIsoParCoords(nDim);
  std::vector<double> opposingElementIsoParCoords(nDim);

  // interpolate nodal values to point-in-elem
  const int sizeOfScalarField = 1;
 
  // pointers to fixed values
  double *p_cNx = &cNx[0];
  double *p_oNx = &oNx[0];

  // nodal fields to gather
  std::vector<double> ws_c_face_scalarQ;
  std::vector<double> ws_o_face_scalarQ;
  std::vector<double> ws_c_elem_scalarQ;
  std::vector<double> ws_o_elem_scalarQ;
  std::vector<double> ws_c_elem_coordinates;
  std::vector<double> ws_o_elem_coordinates;
  std::vector<double> ws_c_diffFluxCoeff;
  std::vector<double> ws_o_diffFluxCoeff;  
  std::vector<double> ws_o_coordinates;

  // master element data
  std::vector<double> ws_c_dndx;
  std::vector<double> ws_o_dndx;
  std::vector<double> ws_c_det_j;
  std::vector<double> ws_o_det_j;
  std::vector <double > ws_c_general_shape_function;
  std::vector <double > ws_o_general_shape_function;

  // deal with state
  ScalarFieldType &scalarQNp1 = scalarQ_->field_of_state(stk::mesh::StateNP1);

  // parallel communicate ghosted entities
  if ( NULL != realm_.nonConformalManager_->nonConformalGhosting_ )
    stk::mesh::communicate_field_data(*(realm_.nonConformalManager_->nonConformalGhosting_), ghostFieldVec_);

  // iterate nonConformalManager's dgInfoVec
  std::vector<NonConformalInfo *>::iterator ii;
  for( ii=realm_.nonConformalManager_->nonConformalInfoVec_.begin();
       ii!=realm_.nonConformalManager_->nonConformalInfoVec_.end(); ++ii ) {

    // extract vector of DgInfo
    std::vector<std::vector<DgInfo *> > &dgInfoVec = (*ii)->dgInfoVec_;
    
    std::vector<std::vector<DgInfo*> >::iterator idg;
    for( idg=dgInfoVec.begin(); idg!=dgInfoVec.end(); ++idg ) {

      std::vector<DgInfo *> &faceDgInfoVec = (*idg);

      // now loop over all the DgInfo objects on this particular exposed face
      for ( size_t k = 0; k < faceDgInfoVec.size(); ++k ) {

        DgInfo *dgInfo = faceDgInfoVec[k];
      
        // extract current/opposing face/element
        stk::mesh::Entity currentFace = dgInfo->currentFace_;
        stk::mesh::Entity opposingFace = dgInfo->opposingFace_;
        stk::mesh::Entity currentElement = dgInfo->currentElement_;
        stk::mesh::Entity opposingElement = dgInfo->opposingElement_;
        const int currentFaceOrdinal = dgInfo->currentFaceOrdinal_;
        const int opposingFaceOrdinal = dgInfo->opposingFaceOrdinal_;
        
        // master element; face and volume
        MasterElement * meFCCurrent = dgInfo->meFCCurrent_; 
        MasterElement * meFCOpposing = dgInfo->meFCOpposing_;
        MasterElement * meSCSCurrent = dgInfo->meSCSCurrent_; 
        MasterElement * meSCSOpposing = dgInfo->meSCSOpposing_;
   
        // local ip, ordinals, etc
        const int currentGaussPointId = dgInfo->currentGaussPointId_;
        currentIsoParCoords = dgInfo->currentIsoParCoords_;
        opposingIsoParCoords = dgInfo->opposingIsoParCoords_;
   
        // mapping from ip to nodes for this ordinal
        const int *ipNodeMap = meSCSCurrent->ipNodeMap(currentFaceOrdinal);

        // extract some master element info
        const int currentNodesPerFace = meFCCurrent->nodesPerElement_;
        const int opposingNodesPerFace = meFCOpposing->nodesPerElement_;
        const int currentNodesPerElement = meSCSCurrent->nodesPerElement_;
        const int opposingNodesPerElement = meSCSOpposing->nodesPerElement_;
       
        // resize some things; matrix related
        const int totalNodes = currentNodesPerElement + opposingNodesPerElement;
        const int lhsSize = totalNodes*totalNodes;
        const int rhsSize = totalNodes;
        lhs.resize(lhsSize);
        rhs.resize(rhsSize);
        scratchIds.resize(rhsSize);
        scratchVals.resize(rhsSize);
        connected_nodes.resize(totalNodes);
        
        // algorithm related; element; dndx will be at a single gauss point...
        ws_c_elem_scalarQ.resize(currentNodesPerElement);
        ws_o_elem_scalarQ.resize(opposingNodesPerElement);
        ws_c_elem_coordinates.resize(currentNodesPerElement*nDim);
        ws_o_elem_coordinates.resize(opposingNodesPerElement*nDim);
        ws_c_dndx.resize(nDim*currentNodesPerElement);
        ws_o_dndx.resize(nDim*opposingNodesPerElement);
        ws_c_det_j.resize(1);
        ws_o_det_j.resize(1);
        
        // algorithm related; face
        ws_c_face_scalarQ.resize(currentNodesPerFace);
        ws_o_face_scalarQ.resize(opposingNodesPerFace);
        ws_c_diffFluxCoeff.resize(currentNodesPerFace);
        ws_o_diffFluxCoeff.resize(opposingNodesPerFace);
        ws_c_general_shape_function.resize(currentNodesPerFace);
        ws_o_general_shape_function.resize(opposingNodesPerFace);
        ws_o_coordinates.resize(opposingNodesPerFace*nDim);

        // pointers
        double *p_lhs = &lhs[0];
        double *p_rhs = &rhs[0];        

        double *p_c_face_scalarQ = &ws_c_face_scalarQ[0];
        double *p_o_face_scalarQ = &ws_o_face_scalarQ[0];
        double *p_c_elem_scalarQ = &ws_c_elem_scalarQ[0];
        double *p_o_elem_scalarQ = &ws_o_elem_scalarQ[0];
        double *p_c_elem_coordinates = &ws_c_elem_coordinates[0];
        double *p_o_elem_coordinates = &ws_o_elem_coordinates[0];
        double *p_c_diffFluxCoeff = &ws_c_diffFluxCoeff[0];
        double *p_o_diffFluxCoeff = &ws_o_diffFluxCoeff[0];
        double *p_o_coordinates= &ws_o_coordinates[0];

        // me pointers
        double *p_c_general_shape_function = &ws_c_general_shape_function[0];
        double *p_o_general_shape_function = &ws_o_general_shape_function[0];
        double *p_c_dndx = &ws_c_dndx[0];
        double *p_o_dndx = &ws_o_dndx[0];
        
        // populate current face_node_ordinals
        const int *c_face_node_ordinals = meSCSCurrent->side_node_ordinals(currentFaceOrdinal);

        // gather current face data
        stk::mesh::Entity const* current_face_node_rels = bulk_data.begin_nodes(currentFace);
        const int current_num_face_nodes = bulk_data.num_nodes(currentFace);
        for ( int ni = 0; ni < current_num_face_nodes; ++ni ) {
          stk::mesh::Entity node = current_face_node_rels[ni];
          // gather...
          p_c_face_scalarQ[ni] = *stk::mesh::field_data(scalarQNp1, node);
          p_c_diffFluxCoeff[ni] = *stk::mesh::field_data(*diffFluxCoeff_, node);
        }
        
        // populate opposing face_node_ordinals
        const int *o_face_node_ordinals = meSCSOpposing->side_node_ordinals(opposingFaceOrdinal);

        // gather opposing face data
        stk::mesh::Entity const* opposing_face_node_rels = bulk_data.begin_nodes(opposingFace);
        const int opposing_num_face_nodes = bulk_data.num_nodes(opposingFace);
        for ( int ni = 0; ni < opposing_num_face_nodes; ++ni ) {
          stk::mesh::Entity node = opposing_face_node_rels[ni];
          // gather; scalar
          p_o_face_scalarQ[ni] = *stk::mesh::field_data(scalarQNp1, node);
          p_o_diffFluxCoeff[ni] = *stk::mesh::field_data(*diffFluxCoeff_, node);
          // gather; vector
          const double *coords = stk::mesh::field_data(*coordinates_, node);
          for ( int i = 0; i < nDim; ++i ) {
            p_o_coordinates[ni*nDim+i] = coords[i];
          }
        }
        
        // gather current element data; sneak in first of connected nodes
        stk::mesh::Entity const* current_elem_node_rels = bulk_data.begin_nodes(currentElement);
        const int current_num_elem_nodes = bulk_data.num_nodes(currentElement);
        for ( int ni = 0; ni < current_num_elem_nodes; ++ni ) {
          stk::mesh::Entity node = current_elem_node_rels[ni];
          // set connected nodes
          connected_nodes[ni] = node;
          // gather; scalar
          p_c_elem_scalarQ[ni] = *stk::mesh::field_data(scalarQNp1, node);
          // gather; vector
          const double *coords = stk::mesh::field_data(*coordinates_, node);
          const int niNdim = ni*nDim;
          for ( int i = 0; i < nDim; ++i ) {
            p_c_elem_coordinates[niNdim+i] = coords[i];
          }
        }

        // gather opposing element data; sneak in second connected nodes
        stk::mesh::Entity const* opposing_elem_node_rels = bulk_data.begin_nodes(opposingElement);
        const int opposing_num_elem_nodes = bulk_data.num_nodes(opposingElement);
        for ( int ni = 0; ni < opposing_num_elem_nodes; ++ni ) {
          stk::mesh::Entity node = opposing_elem_node_rels[ni];
          // set connected nodes
          connected_nodes[ni+current_num_elem_nodes] = node;
          // gather; scalar
          p_o_elem_scalarQ[ni] = *stk::mesh::field_data(scalarQNp1, node);
          // gather; vector
          const double *coords = stk::mesh::field_data(*coordinates_, node);
          const int niNdim = ni*nDim;
          for ( int i = 0; i < nDim; ++i ) {
            p_o_elem_coordinates[niNdim+i] = coords[i];
          }
        }

        // compute opposing normal through master element call, not using oppoing exposed area
        meFCOpposing->general_normal(&opposingIsoParCoords[0], &p_o_coordinates[0], &p_oNx[0]);
        
        // pointer to face data
        const double * c_areaVec = stk::mesh::field_data(*exposedAreaVec_, currentFace);
        const double * ncMassFlowRate = stk::mesh::field_data(*ncMassFlowRate_, currentFace);
       
        double c_amag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double c_axj = c_areaVec[currentGaussPointId*nDim+j];
          c_amag += c_axj*c_axj;
        }
        c_amag = std::sqrt(c_amag);
        
        // now compute normal
        for ( int i = 0; i < nDim; ++i ) {
          p_cNx[i] = c_areaVec[currentGaussPointId*nDim+i]/c_amag;
        }

        // override opposing normal
        if ( useCurrentNormal_ ) {
          for ( int i = 0; i < nDim; ++i )
            p_oNx[i] = -p_cNx[i];
        }

        // project from side to element; method deals with the -1:1 isInElement range to the proper underlying CVFEM range
        meSCSCurrent->sidePcoords_to_elemPcoords(currentFaceOrdinal, 1, &currentIsoParCoords[0], &currentElementIsoParCoords[0]);
        meSCSOpposing->sidePcoords_to_elemPcoords(opposingFaceOrdinal, 1, &opposingIsoParCoords[0], &opposingElementIsoParCoords[0]);
        
        // compute dndx
        double scs_error = 0.0;
        meSCSCurrent->general_face_grad_op(currentFaceOrdinal, &currentElementIsoParCoords[0], 
                                           &p_c_elem_coordinates[0], &p_c_dndx[0], &ws_c_det_j[0], &scs_error);
        meSCSOpposing->general_face_grad_op(opposingFaceOrdinal, &opposingElementIsoParCoords[0], 
                                           &p_o_elem_coordinates[0], &p_o_dndx[0], &ws_o_det_j[0], &scs_error);

        // current diffusive flux
        double currentDiffFluxBip = 0.0;
        for ( int ic = 0; ic < currentNodesPerElement; ++ic ) {
          const int offSetDnDx = ic*nDim; // single intg. point
          const double scalarQIC = p_c_elem_scalarQ[ic];
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_cNx[j];
            const double dndxj = p_c_dndx[offSetDnDx+j];
            currentDiffFluxBip -= dndxj*nxj*scalarQIC;
          }
        }

        // current inverse length scale; can loop over face nodes to avoid "nodesOnFace" array
        double currentInverseLength = 0.0;
        for ( int ic = 0; ic < current_num_face_nodes; ++ic ) {
          const int faceNodeNumber = c_face_node_ordinals[ic];
          const int offSetDnDx = faceNodeNumber*nDim; // single intg. point
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_cNx[j];
            const double dndxj = p_c_dndx[offSetDnDx+j];
            currentInverseLength += dndxj*nxj;
          }
        }

        // opposing flux
        double opposingDiffFluxBip = 0.0;
        for ( int ic = 0; ic < opposingNodesPerElement; ++ic ) {
          const int offSetDnDx = ic*nDim; // single intg. point
          const double scalarQIC = p_o_elem_scalarQ[ic];
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_oNx[j];
            const double dndxj = p_o_dndx[offSetDnDx+j];
            opposingDiffFluxBip -= dndxj*nxj*scalarQIC;
          }
        }

        // opposing inverse length scale; can loop over face nodes to avoid "nodesOnFace" array
        double opposingInverseLength = 0.0;
        for ( int ic = 0; ic < opposing_num_face_nodes; ++ic ) {
          const int faceNodeNumber = o_face_node_ordinals[ic];
          const int offSetDnDx = faceNodeNumber*nDim; // single intg. point
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_oNx[j];
            const double dndxj = p_o_dndx[offSetDnDx+j];
            opposingInverseLength += dndxj*nxj;
          }
        }

        // interpolate face data; current and opposing...
        double currentScalarQBip = 0.0;
        meFCCurrent->interpolatePoint(
          sizeOfScalarField,
          &(dgInfo->currentIsoParCoords_[0]),
          &ws_c_face_scalarQ[0],
          &currentScalarQBip);
        
        double opposingScalarQBip = 0.0;
        meFCOpposing->interpolatePoint(
          sizeOfScalarField,
          &(dgInfo->opposingIsoParCoords_[0]),
          &ws_o_face_scalarQ[0],
          &opposingScalarQBip);

        double currentDiffFluxCoeffBip = 0.0;
        meFCCurrent->interpolatePoint(
          sizeOfScalarField,
          &(dgInfo->currentIsoParCoords_[0]),
          &ws_c_diffFluxCoeff[0],
          &currentDiffFluxCoeffBip);

        double opposingDiffFluxCoeffBip = 0.0;
        meFCOpposing->interpolatePoint(
          sizeOfScalarField,
          &(dgInfo->opposingIsoParCoords_[0]),
          &ws_o_diffFluxCoeff[0],
          &opposingDiffFluxCoeffBip);
                
        // properly scaled diffusive flux
        currentDiffFluxBip *= currentDiffFluxCoeffBip;
        opposingDiffFluxBip *= opposingDiffFluxCoeffBip;

        // zero lhs/rhs
        for ( int p = 0; p < lhsSize; ++p )
          p_lhs[p] = 0.0;
        for ( int p = 0; p < rhsSize; ++p )
          p_rhs[p] = 0.0;

        // save mdot and |mdot|
        const double tmdot = ncMassFlowRate[currentGaussPointId];
        const double abs_tmdot = std::abs(tmdot);

        // compute penalty
        const double penaltyIp 
          = (currentDiffFluxCoeffBip*currentInverseLength + opposingDiffFluxCoeffBip*opposingInverseLength)/2.0;
       
        // non conformal diffusive flux
        const double ncDiffFlux =  (currentDiffFluxBip - opposingDiffFluxBip)/2.0;
       
        // non conformal advection
        const double ncAdv = tmdot*(currentScalarQBip + opposingScalarQBip)/2.0 
          + eta_*abs_tmdot*(currentScalarQBip-opposingScalarQBip)/2.0;
       
        // form residual
        const int nn = ipNodeMap[currentGaussPointId];
        p_rhs[nn] -= ((ncDiffFlux + penaltyIp*(currentScalarQBip-opposingScalarQBip))*c_amag + ncAdv);

        // set-up row for matrix
        const int rowR = nn*totalNodes;
        
        // sensitivities; current face (penalty and advection); use general shape function for this single ip
        const double lhsFacC = penaltyIp*c_amag + (eta_*abs_tmdot + tmdot)/2.0;
        meFCCurrent->general_shape_fcn(1, &currentIsoParCoords[0], &ws_c_general_shape_function[0]);
        for ( int ic = 0; ic < currentNodesPerFace; ++ic ) {
          const int icnn = c_face_node_ordinals[ic];
          const double r = p_c_general_shape_function[ic];
          p_lhs[rowR+icnn] += r*lhsFacC;
        }
        
        // sensitivities; current element (diffusion)
        for ( int ic = 0; ic < currentNodesPerElement; ++ic ) {
          const int offSetDnDx = ic*nDim; // single intg. point
          double lhscd = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_cNx[j];
            const double dndxj = p_c_dndx[offSetDnDx+j];
            lhscd -= dndxj*nxj;
          }
          p_lhs[rowR+ic] += currentDiffFluxCoeffBip*lhscd*c_amag/2.0;
        }

        // sensitivities; opposing face (penalty and advection); use general shape function for this single ip
        const double lhsFacO = penaltyIp*c_amag + (eta_*abs_tmdot - tmdot)/2.0;
        meFCOpposing->general_shape_fcn(1, &opposingIsoParCoords[0], &ws_o_general_shape_function[0]);
        for ( int ic = 0; ic < opposingNodesPerFace; ++ic ) {
          const int icnn = o_face_node_ordinals[ic];
          const double r = p_o_general_shape_function[ic];
          p_lhs[rowR+icnn+currentNodesPerElement] -= r*lhsFacO;
        }

        // sensitivities; opposing element (diffusion)
        for ( int ic = 0; ic < opposingNodesPerElement; ++ic ) {
          const int offSetDnDx = ic*nDim; // single intg. point
          double lhscd = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double nxj = p_oNx[j];
            const double dndxj = p_o_dndx[offSetDnDx+j];
            lhscd -= dndxj*nxj;
          }
          p_lhs[rowR+ic+currentNodesPerElement] -= opposingDiffFluxCoeffBip*lhscd*c_amag/2.0;
        }

        // relax the diagonal term before applying to the matrix
        for (int ir=0; ir < totalNodes; ir++) {
          p_lhs[ir * (totalNodes + 1)] /= relaxFac;
        }

        apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
