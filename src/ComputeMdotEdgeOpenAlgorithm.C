/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include "ComputeMdotEdgeOpenAlgorithm.h"
#include "Algorithm.h"

#include "FieldTypeDef.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "master_element/MasterElement.h"

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
// ComputeMdotEdgeOpenAlgorithm - compute mdot at edges ip
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotEdgeOpenAlgorithm::ComputeMdotEdgeOpenAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    meshMotion_(realm_.does_mesh_move()),
    velocityRTM_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    exposedAreaVec_(NULL),
    openMassFlowRate_(NULL),
    pressureBc_(NULL)
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
  openMassFlowRate_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "open_mass_flow_rate");
  pressureBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, realm_.solutionOptions_->activateOpenMdotCorrection_ ? "pressure" : "pressure_bc");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotEdgeOpenAlgorithm::~ComputeMdotEdgeOpenAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotEdgeOpenAlgorithm::execute()
{
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract noc
  const std::string dofName = "pressure";
  const double nocFac
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // extract global algorithm options, if active
  const double pstabFac = realm_.solutionOptions_->activateOpenMdotCorrection_ 
    ? 0.0
    : 1.0;

  // interpolation for mdot uses nearest node, therefore, n/a

  // set accumulation variables
  double mdotOpen = 0.0;
  size_t mdotOpenIpCount = 0;

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
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

    // size some things that are useful
    const int num_face_nodes = b.topology().num_nodes();
    
    const stk::mesh::Bucket::size_type length   = b.size();

    mdotOpenIpCount += length*num_face_nodes;

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);
      double * mdot = stk::mesh::field_data(*openMassFlowRate_, b, k);

      // extract the connected element to this exposed face; should be single in size!
      stk::mesh::Entity const * face_elem_rels = b.begin_elements(k);
      ThrowAssert( b.num_elements(k) == 1 );

      // get element; its face ordinal number and populate face_node_ordinals
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = b.begin_element_ordinals(k)[0];
      const int *face_node_ordinals = meSCS->side_node_ordinals(face_ordinal);

      // get the relations
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

      for ( int ip = 0; ip < num_face_nodes; ++ip ) {

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int nearestNode = face_node_ordinals[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = elem_node_rels[nearestNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        const double pressureL = *stk::mesh::field_data(*pressure_, nodeL );
        const double pressureR = *stk::mesh::field_data(*pressure_, nodeR );
        const double pressureIp = 0.5*(pressureL + pressureR);

        // nearest nodes
        const double * GpdxR =  stk::mesh::field_data(*Gpdx_, nodeR );
        const double * vrtm =  stk::mesh::field_data(*velocityRTM_, nodeR );
        const double densityR = *stk::mesh::field_data(densityNp1, nodeR );
        const double bcPressure = *stk::mesh::field_data(*pressureBc_, nodeR );

        const double udiagR = *stk::mesh::field_data(*Udiag, nodeR);
        const double projTimeScale = pstabFac / udiagR;

        // offset for bip area vector
        const int faceOffSet = ip*nDim;

        // compute geometry
        double axdx = 0.0;
        double asq = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          const double coordIp = 0.5*(coordR[j] + coordL[j]);
          const double dxj = coordR[j]  - coordIp;
          asq += axj*axj;
          axdx += axj*dxj;
        }

        const double inv_axdx = 1.0/axdx;
        const double rhoBip = densityR;

        // mdot
        double tmdot = -projTimeScale*(bcPressure-pressureIp)*asq*inv_axdx;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[faceOffSet+j];
          const double coordIp = 0.5*(coordR[j] + coordL[j]);
          const double dxj = coordR[j]  - coordIp;
          const double kxj = axj - asq*inv_axdx*dxj;
          const double Gjp = GpdxR[j];
          tmdot += (rhoBip*vrtm[j]+projTimeScale*Gjp)*axj
            - projTimeScale*kxj*Gjp*nocFac;
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
