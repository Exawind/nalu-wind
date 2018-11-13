/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemOpenSolverAlgorithm.h>
#include <EquationSystem.h>
#include <FieldTypeDef.h>
#include <LinearSystem.h>
#include <PecletFunction.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <master_element/MasterElement.h>

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
// AssembleMomentumElemOpenSolverAlgorithm - lhs for momentum open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemOpenSolverAlgorithm::AssembleMomentumElemOpenSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem)
  : SolverAlgorithm(realm, part, eqSystem),
    includeDivU_(realm_.get_divU()),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    velocity_(NULL),
    dudx_(NULL),
    coordinates_(NULL),
    density_(NULL),
    viscosity_(NULL),
    exposedAreaVec_(NULL),
    openMassFlowRate_(NULL),
    velocityBc_(NULL),
    pecletFunction_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  const std::string viscName = realm.is_turbulent()
    ? "effective_viscosity_u" : "viscosity";
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, viscName);
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  velocityBc_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "open_velocity_bc");

  // create the peclet blending function
  pecletFunction_ = eqSystem->create_peclet_function<double>(velocity_->name());
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemOpenSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildFaceElemToNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemOpenSolverAlgorithm::~AssembleMomentumElemOpenSolverAlgorithm()
{
  delete pecletFunction_;
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemOpenSolverAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // extract user advection options (allow to potentially change over time)
  const std::string dofName = "velocity";
  const double alphaUpw = realm_.get_alpha_upw_factor(dofName);
  const double hoUpwind = realm_.get_upw_factor(dofName);
  const bool useShiftedGradOp = realm_.get_shifted_grad_op(dofName);
  const bool skewSymmetric = realm_.get_skew_symmetric(dofName);

  const double relaxFacU = realm_.solutionOptions_->get_relaxation_factor(dofName);

  // one minus flavor..
  const double om_alphaUpw = 1.0-alphaUpw;

  // nearest face entrainment
  const double nfEntrain = realm_.solutionOptions_->nearestFaceEntrain_;
  const double om_nfEntrain = 1.0-nfEntrain;

  // space for LHS/RHS; nodesPerElem*nDim*nodesPerElem*nDim and nodesPerElem*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // ip values; both boundary and opposing surface
  std::vector<double> uBip(nDim);
  std::vector<double> uScs(nDim);
  std::vector<double> uBipExtrap(nDim);
  std::vector<double> uspecBip(nDim);
  std::vector<double> coordBip(nDim);
  std::vector<double> nx(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_uScs = &uScs[0];
  double *p_uBipExtrap = &uBipExtrap[0];
  double *p_uspecBip = &uspecBip[0];
  double *p_coordBip = &coordBip[0];
  double *p_nx = &nx[0];

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_dudx;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_viscosity;
  std::vector<double> ws_bcVelocity;
  // master element
  std::vector<double> ws_face_shape_function;
  std::vector<double> ws_shape_function;
  std::vector<double> ws_adv_face_shape_function;
  std::vector<double> ws_adv_shape_function;
  std::vector<double> ws_dndx;
  std::vector<double> ws_det_j;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

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
    const int numScsIp = meSCS->numIntPoints_;

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->numIntPoints_;


    // resize some things; matrix related
    const int lhsSize = nodesPerElement*nDim*nodesPerElement*nDim;
    const int rhsSize = nodesPerElement*nDim;
    lhs.resize(lhsSize);
    rhs.resize(rhsSize);
    scratchIds.resize(rhsSize);
    scratchVals.resize(rhsSize);
    connected_nodes.resize(nodesPerElement);

    // algorithm related; element
    ws_velocityNp1.resize(nodesPerElement*nDim);
    ws_dudx.resize(nodesPerFace*nDim*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_viscosity.resize(nodesPerFace);
    ws_bcVelocity.resize(nodesPerFace*nDim);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);
    ws_shape_function.resize(numScsIp*nodesPerElement);
    if ( skewSymmetric ) {
      ws_adv_face_shape_function.resize(numScsBip*nodesPerFace);
      ws_adv_shape_function.resize(numScsIp*nodesPerElement);
    }

    ws_dndx.resize(nDim*numScsBip*nodesPerElement);
    ws_det_j.resize(numScsBip);

    // pointers
    double *p_lhs = &lhs[0];
    double *p_rhs = &rhs[0];
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_dudx = &ws_dudx[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_viscosity = &ws_viscosity[0];
    double *p_bcVelocity = &ws_bcVelocity[0];
    double *p_face_shape_function = &ws_face_shape_function[0];
    double *p_shape_function = &ws_shape_function[0];
    double *p_adv_face_shape_function =  skewSymmetric ? &ws_adv_face_shape_function[0] : &ws_face_shape_function[0];;
    double *p_adv_shape_function = skewSymmetric ? &ws_adv_shape_function[0] : &ws_shape_function[0];
   
    double *p_dndx = &ws_dndx[0];

    // shape function
    meSCS->shape_fcn(&p_shape_function[0]);
    meFC->shape_fcn(&p_face_shape_function[0]);
    if ( skewSymmetric ) {
      meSCS->shifted_shape_fcn(&p_adv_shape_function[0]);
      meFC->shifted_shape_fcn(&p_adv_face_shape_function[0]);
    }

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
        p_viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);

        // gather vectors
        double * uspec = stk::mesh::field_data(*velocityBc_, node);
        double * Gjui = stk::mesh::field_data(*dudx_, node);
        const int niNdim = ni*nDim;
        const int row_p_dudx = niNdim*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_bcVelocity[niNdim+i] = uspec[i];
          // gather tensor
          const int row_dudx = i*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_dudx[row_p_dudx+row_dudx+j] = Gjui[row_dudx+j];
          }
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      // mapping from ip to nodes for this ordinal; 
      const int *ipNodeMap = meSCS->ipNodeMap(face_ordinal); // use with elem_node_rels
      const int *faceIpNodeMap = meFC->ipNodeMap(); // use with face_node_rels

      //==========================================
      // gather nodal data off of element
      //==========================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        // set connected nodes
        connected_nodes[ni] = node;
        // gather vectors
        double * uNp1 = stk::mesh::field_data(velocityNp1, node);
        double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_velocityNp1[offSet+j] = uNp1[j];
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // compute dndx
      double scs_error = 0.0;
      if ( useShiftedGradOp )
        meSCS->shifted_face_grad_op(1, face_ordinal, &p_coordinates[0], &p_dndx[0], &ws_det_j[0], &scs_error);
      else
        meSCS->face_grad_op(1, face_ordinal, &p_coordinates[0], &p_dndx[0], &ws_det_j[0], &scs_error);
      
      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int nearestNode = ipNodeMap[ip];
        const int opposingScsIp = meSCS->opposingFace(face_ordinal,ip);
        const int localFaceNode = faceIpNodeMap[ip];

        // offset for bip area vector and types of shape function
        const int faceOffSet = ip*nDim;
        const int offSetSF_face = ip*nodesPerFace;
        const int offSetSF_elem = opposingScsIp*nodesPerElement;

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // zero out vector quantities
        double asq = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          p_uBip[j] = 0.0;
          p_uScs[j] = 0.0;
          p_uspecBip[j] = 0.0;
          p_coordBip[j] = 0.0;
          const double axj = areaVec[faceOffSet+j];
          asq += axj*axj;
        }
        const double amag = std::sqrt(asq);

        // interpolate to bip
        double viscBip = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          const double rAdv = p_adv_face_shape_function[offSetSF_face+ic];
          viscBip += r*p_viscosity[ic];
          const int offSetFN = ic*nDim;
          const int nn = face_node_ordinals[ic];
          const int offSetEN = nn*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uspecBip[j] += rAdv*p_bcVelocity[offSetFN+j];
            p_uBip[j] += rAdv*p_velocityNp1[offSetEN+j];
            p_coordBip[j] += rAdv*p_coordinates[offSetEN+j];
          }
        }

        // data at interior opposing face
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double rAdv = p_adv_shape_function[offSetSF_elem+ic];
          const int offSet = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uScs[j] += rAdv*p_velocityNp1[offSet+j];
          }
        }

        // Peclet factor; along the edge is fine
        const double densL   = *stk::mesh::field_data(densityNp1, nodeL);
        const double densR   = *stk::mesh::field_data(densityNp1, nodeR);
        const double viscL   = *stk::mesh::field_data(*viscosity_, nodeL);
        const double viscR   = *stk::mesh::field_data(*viscosity_, nodeR);
        const double *uNp1R  =  stk::mesh::field_data(velocityNp1, nodeR);
        const double *vrtmL  =  stk::mesh::field_data(*velocityRTM_, nodeL);
        const double *vrtmR  =  stk::mesh::field_data(*velocityRTM_, nodeR);
        
        const double *coordL =  stk::mesh::field_data(*coordinates_, nodeL);
        const double *coordR =  stk::mesh::field_data(*coordinates_, nodeR);

        double udotx = 0.0;
        const int row_p_dudxR = localFaceNode*nDim*nDim; // tricky here with localFaceNode
        for ( int i = 0; i < nDim; ++i ) {
          const double dxi = coordR[i]  - coordL[i];
          udotx += 0.5*dxi*(vrtmL[i] + vrtmR[i]);
          p_nx[i] = areaVec[faceOffSet+i]/amag;
          // extrapolation
          double duR = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            double dxj = coordBip[j] - coordR[j];
            duR += dxj*p_dudx[row_p_dudxR+i*nDim+j]*hoUpwind;
          }
          p_uBipExtrap[i] = uNp1R[i] + duR;
        }

        const double diffIp = 0.5*(viscL/densL + viscR/densR);
        const double pecfac = pecletFunction_->execute(std::abs(udotx)/(diffIp+small));
        const double om_pecfac = 1.0-pecfac;

        //================================
        // advection first
        //================================
        const double tmdot = mdot[ip];

        // advection; leaving the domain
        if ( tmdot > 0.0 ) {

          for ( int i = 0; i < nDim; ++i ) {

            const int indexR = nearestNode*nDim + i;
            const int rowR = indexR*nodesPerElement*nDim;

            // central
            const double uiIp = p_uBip[i];

            // upwind
            const double uiUpwind = alphaUpw*p_uBipExtrap[i] + om_alphaUpw*uiIp;

            // total advection; pressure contribution in time expression
            const double aflux = tmdot*(pecfac*uiUpwind+om_pecfac*uiIp);

            p_rhs[indexR] -= aflux;

            // upwind lhs
            p_lhs[rowR+indexR] += tmdot*pecfac*alphaUpw;

            // central part
            const double fac = tmdot*(pecfac*om_alphaUpw+om_pecfac);
            for ( int ic = 0; ic < nodesPerFace; ++ic ) {
              const int nn = face_node_ordinals[ic];
              p_lhs[rowR+nn*nDim+i] += p_adv_face_shape_function[offSetSF_face+ic]*fac;
            }
          }
        }
        else {
          // extrainment
          double ubipnx = 0.0;
          double ubipExtrapnx = 0.0;
          double uscsnx = 0.0;
          double uspecbipnx = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double nj = p_nx[j];
            ubipnx += p_uBip[j]*nj;
            ubipExtrapnx += p_uBipExtrap[j]*nj;
            uscsnx += p_uScs[j]*nj;
            uspecbipnx += p_uspecBip[j]*nj;
          }

          for ( int i = 0; i < nDim; ++i ) {
            const int indexR = nearestNode*nDim + i;
            const int rowR = indexR*nodesPerElement*nDim;
            const double nxi = p_nx[i];

            // total advection; with tangeant entrain
            const double aflux = tmdot*(pecfac*ubipExtrapnx+om_pecfac*
                                        (nfEntrain*ubipnx + om_nfEntrain*uscsnx))*nxi
              + tmdot*(p_uspecBip[i] - uspecbipnx*nxi);

            p_rhs[indexR] -= aflux;

            // upwind and central
            for ( int j = 0; j < nDim; ++j ) {
              const double nxinxj = nxi*p_nx[j];

              // upwind
              p_lhs[rowR+nearestNode*nDim+j] += tmdot*pecfac*alphaUpw*nxinxj;

              // central part; exposed face
              double fac = tmdot*(pecfac*om_alphaUpw+om_pecfac*nfEntrain)*nxinxj;
              for ( int ic = 0; ic < nodesPerFace; ++ic ) {
                const int nn = face_node_ordinals[ic];
                p_lhs[rowR+nn*nDim+j] += p_adv_face_shape_function[offSetSF_face+ic]*fac;
              }

              // central part; scs face
              fac = tmdot*om_pecfac*om_nfEntrain*nxinxj;
              for ( int ic = 0; ic < nodesPerElement; ++ic ) {
                p_lhs[rowR+ic*nDim+j] += p_adv_shape_function[offSetSF_elem+ic]*fac;
              }

            }
          }
        }
        
        //================================
        // diffusion second
        //================================
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;

          for ( int j = 0; j < nDim; ++j ) {

            const double axj = areaVec[faceOffSet+j];
            const double dndxj = p_dndx[offSetDnDx+j];
            const double uxj = p_velocityNp1[ic*nDim+j];

            const double divUstress = 2.0/3.0*viscBip*dndxj*uxj*axj*includeDivU_;

            for ( int i = 0; i < nDim; ++i ) {

              // matrix entries
              int indexR = nearestNode*nDim + i;
              int rowR = indexR*nodesPerElement*nDim;

              const double dndxi = p_dndx[offSetDnDx+i];
              const double uxi = p_velocityNp1[ic*nDim+i];
              const double nxi = p_nx[i];
              const double om_nxinxi = 1.0-nxi*nxi;

              // -mu*dui/dxj*Aj(1.0-nini); sneak in divU (explicit)
              double lhsfac = -viscBip*dndxj*axj*om_nxinxi;
              p_lhs[rowR+ic*nDim+i] += lhsfac;
              p_rhs[indexR] -= lhsfac*uxi + divUstress*om_nxinxi;

              // -mu*duj/dxi*Aj(1.0-nini)
              lhsfac = -viscBip*dndxi*axj*om_nxinxi;
              p_lhs[rowR+ic*nDim+j] += lhsfac;
              p_rhs[indexR] -= lhsfac*uxj;

              // now we need the -nx*ny*Fy - nx*nz*Fz part
              for ( int l = 0; l < nDim; ++l ) {

                if ( i != l ) {
                  const double nxinxl = nxi*p_nx[l];
                  const double uxl = p_velocityNp1[ic*nDim+l];
                  const double dndxl = p_dndx[offSetDnDx+l];

                  // +ni*nl*mu*dul/dxj*Aj; sneak in divU (explicit)
                  lhsfac = viscBip*dndxj*axj*nxinxl;
                  p_lhs[rowR+ic*nDim+l] += lhsfac;
                  p_rhs[indexR] -= lhsfac*uxl + divUstress*nxinxl;

                  // +ni*nl*mu*duj/dxl*Aj
                  lhsfac = viscBip*dndxl*axj*nxinxl;
                  p_lhs[rowR+ic*nDim+j] += lhsfac;
                  p_rhs[indexR] -= lhsfac*uxj;
                }
              }
            }
          }
        }
      }

      // relax the diagonal term before applying to the matrix
      const int npeNdim = nodesPerElement*nDim;
      for (int ir=0; ir < npeNdim; ir++) {
        p_lhs[ir * (npeNdim + 1)] /= relaxFacU;
      }

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);

    }
  }
}

} // namespace nalu
} // namespace Sierra
