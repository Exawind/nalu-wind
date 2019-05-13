/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include "ComputeMdotInflowAlgorithm.h"
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
// ComputeMdotInflowAlgorithm - mdot continuity inflow bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotInflowAlgorithm::ComputeMdotInflowAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  bool useShifted)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    velocityBC_(NULL),
    densityBC_(NULL),
    exposedAreaVec_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocityBC_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.solutionOptions_->activateOpenMdotCorrection_ 
                                                     ? "velocity_bc" : "cont_velocity_bc");
  // variable density will need density as a function of user inflow conditions
  densityBC_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotInflowAlgorithm::~ComputeMdotInflowAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotInflowAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // set accumulation variables
  double mdotInflow = 0.0;

  // nodal fields to gather; gather everything other than what we are assembling
  std::vector<double> ws_densityBC;
  std::vector<double> ws_velocityBC;

  // geometry related to populate
  std::vector<double> ws_shape_function;

  // ip data
  std::vector<double>uIp(nDim);
  std::vector<double>rho_uIp(nDim);
  double *p_uIp = &uIp[0];
  double *p_rho_uIp = &rho_uIp[0];

  ScalarFieldType &densityNp1 = densityBC_->field_of_state(stk::mesh::StateNP1);

  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);
  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element specifics
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // algorithm related
    ws_densityBC.resize(nodesPerFace);
    ws_velocityBC.resize(nodesPerFace*nDim);
    ws_shape_function.resize(numScsBip*nodesPerFace);

    // pointers
    double *p_density = &ws_densityBC[0];
    double *p_velocity = &ws_velocityBC[0];
    double *p_shape_function = &ws_shape_function[0];

    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_shape_function[0]);
    else
      meFC->shape_fcn(&p_shape_function[0]);

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // face data
      double * areaVec = stk::mesh::field_data(*exposedAreaVec_, b, k);

      // face node relations for nodal gather
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);

      int num_nodes = b.num_nodes(k);
      for ( int ni = 0; ni < num_nodes; ++ni ) {

        // get the node and form connected_node
        stk::mesh::Entity node = face_node_rels[ni];

        // velocity at nodes
        double * uBc = stk::mesh::field_data(*velocityBC_, node);

        // gather scalar
        p_density[ni] = *stk::mesh::field_data(densityNp1, node);

        // gather vectors
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_velocity[offSet+j] = uBc[j];
        }
      }

      for ( int ip = 0; ip < numScsBip; ++ip ) {

        // interpolate to scs point; operate on saved off ws_field
        for (int j=0; j < nDim; ++j ) {
          p_uIp[j] = 0.0;
          p_rho_uIp[j] = 0.0;
        }

        double rhoIp = 0.0;
        const int offSet = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          const double r = p_shape_function[offSet+ic];
          const double rhoIC = p_density[ic];
          rhoIp += r*rhoIC;
          for ( int j = 0; j < nDim; ++j ) {
            p_uIp[j] += r*p_velocity[ic*nDim+j];
            p_rho_uIp[j] += r*rhoIC*p_velocity[ic*nDim+j];
          }
        }

        double mdot = 0.0;
        for ( int j=0; j < nDim; ++j ) {
          mdot += (interpTogether*p_rho_uIp[j] + om_interpTogether*rhoIp*p_uIp[j])*areaVec[ip*nDim+j];
        }
        
        mdotInflow += mdot;
      }
    }
  }
  // scatter back to solution options
  realm_.solutionOptions_->mdotAlgInflow_ += mdotInflow;
}

} // namespace nalu
} // namespace Sierra
