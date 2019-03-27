/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include "ComputeMdotElemOpenPenaltyAlgorithm.h"
#include "Algorithm.h"

#include "FieldTypeDef.h"
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
// ComputeMdotElemOpenPenaltyAlgorithm - mdot continuity open bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemOpenPenaltyAlgorithm::ComputeMdotElemOpenPenaltyAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    velocityRTM_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    exposedAreaVec_(NULL),
    pressureBc_(NULL),
    interpTogether_(realm_.get_mdot_interp()),
    om_interpTogether_(1.0 - interpTogether_),
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftedGradOp_(realm_.get_shifted_grad_op("pressure")),
    stabFac_(2.0)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( realm_.does_mesh_move() )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  pressureBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, realm_.solutionOptions_->activateOpenMdotCorrection_ ? "pressure" : "pressure_bc");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemOpenPenaltyAlgorithm::~ComputeMdotElemOpenPenaltyAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotElemOpenPenaltyAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  
  // extract global algorithm options, if active
  const double pstabFac = realm_.solutionOptions_->activateOpenMdotCorrection_ 
    ? 0.0
    : 1.0;

  // ip values; both boundary and opposing surface
  std::vector<double> uBip(nDim);
  std::vector<double> rho_uBip(nDim);
  std::vector<double> GpdxBip(nDim);
  std::vector<double> dpdxBip(nDim);

  // pointers to fixed values
  double *p_uBip = &uBip[0];
  double *p_rho_uBip = &rho_uBip[0];
  double *p_GpdxBip = &GpdxBip[0];
  double *p_dpdxBip = &dpdxBip[0];

  // nodal fields to gather
  std::vector<double> ws_coordinates;
  std::vector<double> ws_face_pressure;
  std::vector<double> ws_pressure;
  std::vector<double> ws_vrtm;
  std::vector<double> ws_Gpdx;
  std::vector<double> ws_density;
  std::vector<double> ws_bcPressure;

  // master element
  std::vector<double> ws_face_shape_function;
  std::vector<double> ws_dndx; 
  std::vector<double> ws_det_j;

  // time step; scale projection time scale by pstabFac (no divide by here)
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1*pstabFac;

  // set accumulation variables
  double mdotOpen = 0.0;
  size_t mdotOpenIpCount = 0;

  // deal with state
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
   
    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->numIntPoints_;

    // algorithm related; element (exposed face and element)
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_face_pressure.resize(nodesPerFace);
    ws_pressure.resize(nodesPerElement);
    ws_vrtm.resize(nodesPerFace*nDim);
    ws_Gpdx.resize(nodesPerFace*nDim);
    ws_density.resize(nodesPerFace);
    ws_bcPressure.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);
    ws_dndx.resize(nDim*numScsBip*nodesPerElement);
    ws_det_j.resize(numScsBip);    

    // pointers
    double *p_coordinates = &ws_coordinates[0];
    double *p_face_pressure = &ws_face_pressure[0];
    double *p_pressure = &ws_pressure[0];
    double *p_vrtm = &ws_vrtm[0];
    double *p_Gpdx = &ws_Gpdx[0];
    double *p_density = &ws_density[0];
    double *p_bcPressure = &ws_bcPressure[0];
    double *p_face_shape_function = &ws_face_shape_function[0];
    double *p_dndx = &ws_dndx[0];

    // shape functions; boundary
    if ( shiftMdot_ )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);
    
    const stk::mesh::Bucket::size_type length   = b.size();

    mdotOpenIpCount += length*numScsBip;

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

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
        p_face_pressure[ni]    = *stk::mesh::field_data(*pressure_, node);
        p_bcPressure[ni] = *stk::mesh::field_data(*pressureBc_, node);
        p_density[ni]    = *stk::mesh::field_data(densityNp1, node);

        // gather vectors
        double * vrtm = stk::mesh::field_data(*velocityRTM_, node);
        double * Gjp = stk::mesh::field_data(*Gpdx_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_vrtm[offSet+j] = vrtm[j];
          p_Gpdx[offSet+j] = Gjp[j];
        }
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      double * mdot = stk::mesh::field_data(*openMassFlowRate_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      //======================================
      // gather nodal data off of element
      //======================================
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);
      int num_nodes = bulk_data.num_nodes(element);
      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];

        // gather scalars
        p_pressure[ni]    = *stk::mesh::field_data(*pressure_, node);

        // gather vectors
        double * coords = stk::mesh::field_data(*coordinates_, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // compute dndx
      double scs_error = 0.0;
      if ( shiftedGradOp_ )
        meSCS->shifted_face_grad_op(1, face_ordinal, &p_coordinates[0], &p_dndx[0], &ws_det_j[0], &scs_error);
      else
        meSCS->face_grad_op(1, face_ordinal, &p_coordinates[0], &p_dndx[0], &ws_det_j[0], &scs_error);

      // loop over boundary ips
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        // zero out vector quantities; form aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          p_uBip[j] = 0.0;
          p_rho_uBip[j] = 0.0;
          p_GpdxBip[j] = 0.0;
          p_dpdxBip[j] = 0.0;
          const double axj = areaVec[ip*nDim+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // form L^-1
        double inverseLengthScale = 0.0;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const int faceNodeNumber = face_node_ordinals[ic];
          const int offSetDnDx = nDim*nodesPerElement*ip + faceNodeNumber*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            inverseLengthScale += p_dndx[offSetDnDx+j]*areaVec[ip*nDim+j];
          }
        }        
        inverseLengthScale /= aMag;

        // interpolate to bip
        double pBip = 0.0;
        double pbcBip = 0.0;
        double rhoBip = 0.0;
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          const double rhoIC = p_density[ic];
          rhoBip += r*rhoIC;
          pBip += r*p_face_pressure[ic];
          pbcBip += r*p_bcPressure[ic];
          const int icNdim = ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_uBip[j] += r*p_vrtm[icNdim+j];
            p_rho_uBip[j] += r*rhoIC*p_vrtm[icNdim+j];
            p_GpdxBip[j] += r*p_Gpdx[icNdim+j];
          }
        }

        // form dpdxBip
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          const double pIc = p_pressure[ic];
          for ( int j = 0; j < nDim; ++j ) {
            p_dpdxBip[j] += p_dndx[offSetDnDx+j]*pIc;
          }
        }
     
        // form mdot; rho*uj*Aj - projT*(dpdxj - Gjp)*Aj + stabFac*projTimeScale*invL*(pBip - pbcBip)*aMag
        double tmdot = stabFac_*projTimeScale*inverseLengthScale*(pBip - pbcBip)*aMag*pstabFac;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[ip*nDim+j];
          tmdot += (interpTogether_*p_rho_uBip[j] + om_interpTogether_*rhoBip*p_uBip[j]
                    - projTimeScale*(p_dpdxBip[j] - p_GpdxBip[j])*pstabFac)*axj;
        }
        
        // scatter to mdot and accumulate
        mdot[ip] = tmdot;
        mdotOpen += tmdot;
      }
    }
  }
  // scatter back to solution options; not thread safe
  realm_.solutionOptions_->mdotAlgOpen_ += mdotOpen;
  realm_.solutionOptions_->mdotAlgOpenIpCount_ += mdotOpenIpCount;
}

} // namespace nalu
} // namespace Sierra
