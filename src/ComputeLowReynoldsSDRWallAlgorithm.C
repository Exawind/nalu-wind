/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeLowReynoldsSDRWallAlgorithm.h>
#include <Algorithm.h>

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

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeLowReynoldsSDRWallAlgorithm - low Re omega at wall bc; 
//                                      6*nu/(beta*yp^2)
// compute yp locally
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeLowReynoldsSDRWallAlgorithm::ComputeLowReynoldsSDRWallAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool &useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    betaOne_(realm.get_turb_model_constant(TM_betaOne)),
    wallFactor_(realm.get_turb_model_constant(TM_SDRWallFactor))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  sdrBc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "wall_model_sdr_bc");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_sdr");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeLowReynoldsSDRWallAlgorithm::~ComputeLowReynoldsSDRWallAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeLowReynoldsSDRWallAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // nodal fields to gather
  std::vector<double> ws_density;
  std::vector<double> ws_viscosity;

  // master element
  std::vector<double> ws_face_shape_function;

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

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

    // face master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = b.topology().num_nodes();
    const int numScsBip = meFC->numIntPoints_;

    // mapping from ip to nodes for this ordinal
    const int *faceIpNodeMap = meFC->ipNodeMap();

    // algorithm related; element
    ws_density.resize(nodesPerFace);
    ws_viscosity.resize(nodesPerFace);
    ws_face_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_density = &ws_density[0];
    double *p_viscosity = &ws_viscosity[0];
    double *p_face_shape_function = &ws_face_shape_function[0];

    // shape functions
    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_face_shape_function[0]);
    else
      meFC->shape_fcn(&p_face_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();

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
        p_density[ni]    = *stk::mesh::field_data(densityNp1, node);
        p_viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);
      }

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulk_data.begin_elements(face);
      ThrowAssert( bulk_data.num_elements(face) == 1 );

      // get element; its face ordinal number
      stk::mesh::Entity element = face_elem_rels[0];
      const int face_ordinal = bulk_data.begin_element_ordinals(face)[0];

      // get the relations off of element
      stk::mesh::Entity const * elem_node_rels = bulk_data.begin_nodes(element);

      // loop over face nodes
      for ( int ip = 0; ip < numScsBip; ++ip ) {

        const int offSetAveraVec = ip*nDim;

        const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);
        const int localFaceNode = faceIpNodeMap[ip];

        // left and right nodes; right is on the face; left is the opposing node
        stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
        stk::mesh::Entity nodeR = face_node_rels[localFaceNode];

        // extract nodal fields
        const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
        const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

        // aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[offSetAveraVec+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // interpolate to bip
        double rhoBip = 0.0;
        double muBip = 0.0;
        const int offSetSF_face = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_face_shape_function[offSetSF_face+ic];
          rhoBip += r*p_density[ic];
          muBip += r*p_viscosity[ic];
        }
        const double nuBip = muBip/rhoBip;

        // determine yp (approximated by 1/4 distance along edge)
        double ypbip = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double nj = areaVec[offSetAveraVec+j]/aMag;
          const double ej = 0.25*(coordR[j] - coordL[j]);
          ypbip += nj*ej*nj*ej;
        }
        ypbip = std::sqrt(ypbip);

        // compute low Re wall sdr
        const double lowReSdr = wallFactor_*6.0*nuBip/betaOne_/ypbip/ypbip;

        // assemble to nodal quantities; will normalize and assemble in driver
        double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nodeR );
        double * sdrBc = stk::mesh::field_data(*sdrBc_, nodeR );

        *assembledWallArea += aMag;
        *sdrBc += lowReSdr*aMag;
      }
    }
  }
}


} // namespace nalu
} // namespace Sierra
