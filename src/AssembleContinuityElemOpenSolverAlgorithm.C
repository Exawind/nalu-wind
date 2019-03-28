/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include "AssembleContinuityElemOpenSolverAlgorithm.h"
#include "EquationSystem.h"
#include "FieldTypeDef.h"
#include "LinearSystem.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"
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
// AssembleContinuityElemOpenSolverAlgorithm - lhs for continuity open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleContinuityElemOpenSolverAlgorithm::AssembleContinuityElemOpenSolverAlgorithm(
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
    exposedAreaVec_(NULL),
    pressureBc_(NULL),
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftPoisson_(realm_.get_shifted_grad_op("pressure")),
    reducedSensitivities_(realm_.get_cvfem_reduced_sens_poisson())
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
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  pressureBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, realm_.solutionOptions_->activateOpenMdotCorrection_ ? "pressure" : "pressure_bc");

  // Implementation details: code is designed to manage the following
  // When shiftPoisson_ is TRUE, reducedSensitivities_ is enforced to be TRUE
  // However, shiftPoisson_ can be FALSE while reducedSensitivities_ is TRUE
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleContinuityElemOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  
  // extract noc
  const std::string dofName = "pressure";
  const double includeNOC 
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // Classic Nalu projection timescale
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double tauScale = dt / gamma1;

  // extract global algorithm options, if active
  const double mdotCorrection = realm_.solutionOptions_->activateOpenMdotCorrection_ 
    ? realm_.solutionOptions_->mdotAlgOpenCorrection_
    : 0.0;
  const double pstabFac = realm_.solutionOptions_->activateOpenMdotCorrection_ 
    ? 0.0
    : 1.0;

  // space for LHS/RHS; nodesPerElem*nodesPerElem and nodesPerElem
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // ip values; both boundary and opposing surface
  std::vector<double> uBip(nDim);
  std::vector<double> rho_uBip(nDim);
  std::vector<double> GpdxBip(nDim);
  std::vector<double> coordBip(nDim);
  std::vector<double> coordScs(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_rho_uBip = &rho_uBip[0];
  double *p_GpdxBip = &GpdxBip[0];
  double *p_coordBip = &coordBip[0];
  double *p_coordScs = &coordScs[0];

  // nodal fields to gather
  std::vector<double> ws_coordinates;
  std::vector<double> ws_pressure;
  std::vector<double> ws_vrtm;
  std::vector<double> ws_Gpdx;
  std::vector<double> ws_density;
  std::vector<double> ws_udiag;
  std::vector<double> ws_bcPressure;
  // master element
  std::vector<double> ws_shape_function;
  std::vector<double> ws_shape_function_lhs;
  std::vector<double> ws_face_shape_function;

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  ScalarFieldType* Udiag = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "momentum_diag");

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
    const int numScsIp = meSCS->num_integration_points();

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->num_integration_points();


    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nodesPerElement;
    const int rhsSize = nodesPerElement;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // algorithm related; element
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_pressure.resize(nodesPerElement);
    ws_vrtm.resize(nodesPerFace*nDim);
    ws_Gpdx.resize(nodesPerFace*nDim);
    ws_density.resize(nodesPerFace);
    ws_udiag.resize(nodesPerFace);
    ws_bcPressure.resize(nodesPerFace);
    ws_shape_function.resize(numScsIp*nodesPerElement);
    ws_shape_function_lhs.resize(numScsIp*nodesPerElement);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_pressure = &ws_pressure[0];
    double *p_vrtm = &ws_vrtm[0];
    double *p_Gpdx = &ws_Gpdx[0];
    double *p_density = &ws_density[0];
    double *p_udiag = &ws_udiag[0];
    double *p_bcPressure = &ws_bcPressure[0];
    double *p_shape_function = &ws_shape_function[0];
    double *p_shape_function_lhs = shiftPoisson_ ? &ws_shape_function[0] : reducedSensitivities_ ? &ws_shape_function_lhs[0] : &ws_shape_function[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions; interior
    if ( shiftPoisson_ )
      meSCS->shifted_shape_fcn(&p_shape_function[0]);
    else
      meSCS->shape_fcn(&p_shape_function[0]);

    if ( !shiftPoisson_ && reducedSensitivities_ )
      meSCS->shifted_shape_fcn(&p_shape_function_lhs[0]);

    // shape functions; boundary
    if ( shiftMdot_ )
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
      int num_face_nodes = bulk_data.num_nodes(face);
      // sanity check on num nodes
      ThrowAssert( num_face_nodes == nodesPerFace );
      for ( int ni = 0; ni < num_face_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];

        // gather scalars
        p_density[ni] = *stk::mesh::field_data(densityNp1, node);
        p_bcPressure[ni] = *stk::mesh::field_data(*pressureBc_, node);
        p_udiag[ni] = *stk::mesh::field_data(*Udiag, node);

        // gather vectors
        const double * vrtm = stk::mesh::field_data(*velocityRTM_, node);
        const double * Gjp = stk::mesh::field_data(*Gpdx_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_vrtm[offSet+j] = vrtm[j];
          p_Gpdx[offSet+j] = Gjp[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::ConnectivityOrdinal* face_elem_ords = bulk_data.begin_element_ordinals(face);
      const int face_ordinal = face_elem_ords[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      // mapping from ip to nodes for this ordinal
      const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal);

      //======================================
      // gather nodal data off of element
      //======================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // set connected nodes
        connected_nodes[ni] = node;

        // gather scalars
        p_pressure[ni] = *stk::mesh::field_data(*pressure_, node);

        // gather vectors
        const double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int nearestNode = ipNodeMap[ip];
        const int opposingScsIp = meSCS->opposingFace(face_ordinal,ip);

        // zero out vector quantities
        for ( int j = 0; j < nDim; ++j ) {
          p_uBip[j] = 0.0;
          p_rho_uBip[j] = 0.0;
          p_GpdxBip[j] = 0.0;
          p_coordBip[j] = 0.0;
          p_coordScs[j] = 0.0;
        }
        double rhoBip = 0.0;
        double projTimeScaleBip = 0.0;

        // interpolate to bip
        double pBip = 0.0;
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const int fn = face_node_ordinals[ic];
          const double r = p_face_shape_function[offSetSF_face+ic];
          const double rhoIC = p_density[ic];
          const double udiagInvIC = 1.0 / p_udiag[ic];
          rhoBip += r*rhoIC;
          pBip += r*p_bcPressure[ic];
          projTimeScaleBip += r * udiagInvIC;
          const int offSetFN = ic*nDim;
          const int offSetEN = fn*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_vrtm[offSetFN+j];
            p_rho_uBip[j] += r*rhoIC*p_vrtm[offSetFN+j];
            p_GpdxBip[j] += r*p_Gpdx[offSetFN+j] * udiagInvIC;
            p_coordBip[j] += r*p_coordinates[offSetEN+j];
          }
        }

        // data at interior opposing face
        double pScs = 0.0;
        const int offSetSF_elem = opposingScsIp*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function[offSetSF_elem+ic];
          pScs += r*p_pressure[ic];
          const int offSet = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_coordScs[j] += r*p_coordinates[offSet+j];
          }
        }

        // form axdx, asq and mdot (without dp/dn or noc)
        double asq = 0.0;
        double axdx = 0.0;
        double mdot = -mdotCorrection;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = p_coordBip[j] - p_coordScs[j];
          const double axj = areaVec[ip*nDim+j];
          asq += axj*axj;
          axdx += axj*dxj;
          mdot += (interpTogether*p_rho_uBip[j] + om_interpTogether*rhoBip*p_uBip[j] 
                   + p_GpdxBip[j]*pstabFac)*axj;
        }
	
        const double inv_axdx = 1.0/axdx;
	
        // deal with noc
        double noc = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double dxj = p_coordBip[j] - p_coordScs[j];
          const double axj = areaVec[ip*nDim+j];
          const double kxj = axj - asq*inv_axdx*dxj; // NOC
          noc += kxj*p_GpdxBip[j];
        }

        // lhs for pressure system
        int rowR = nearestNode*nodesPerElement;

        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function_lhs[offSetSF_elem+ic];
          p_lhs[rowR+ic] += r*asq*inv_axdx*pstabFac * projTimeScaleBip / tauScale;
        }
        
        // final mdot
        mdot += -projTimeScaleBip*((pBip-pScs)*asq*inv_axdx*pstabFac + noc*includeNOC*pstabFac);

        // residual
        p_rhs[nearestNode] -= mdot / tauScale;
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
