/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeMdotElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeMdotElemAlgorithm - interior mdor for elem continuity
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemAlgorithm::ComputeMdotElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const bool assembleMdotToEdge)
  : Algorithm(realm, part),
    meshMotion_(realm_.does_mesh_move()),
    assembleMdotToEdge_(assembleMdotToEdge),
    velocityRTM_(NULL),
    Gpdx_(NULL),
    coordinates_(NULL),
    pressure_(NULL),
    density_(NULL),
    massFlowRate_(NULL),
    edgeMassFlowRate_(NULL),
    shiftMdot_(realm_.get_cvfem_shifted_mdot()),
    shiftPoisson_(realm_.get_shifted_grad_op("pressure"))
{
   // extract fields; nodal
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  if ( meshMotion_ )
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_rtm");
  else
    velocityRTM_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  Gpdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pressure_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  massFlowRate_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "mass_flow_rate_scs");

  if ( assembleMdotToEdge_ ) {
    // check to make sure edges are active
    if (!realm_.realmUsesEdges_ )
      throw std::runtime_error("Edges need to be activated for mixed edge/scalar; element/cont");
    edgeMassFlowRate_ = meta_data.get_field<ScalarFieldType>(stk::topology::EDGE_RANK, "mass_flow_rate");
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMdotElemAlgorithm::~ComputeMdotElemAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotElemAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // deal with interpolation procedure
  const double interpTogether = realm_.get_mdot_interp();
  const double om_interpTogether = 1.0-interpTogether;

  // nodal fields to gather
  std::vector<double> ws_vrtm;
  std::vector<double> ws_Gpdx;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_pressure;
  std::vector<double> ws_density;
  std::vector<double> ws_udiag;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_shape_function;

  // integration point data that depends on size
  std::vector<double> uIp(nDim);
  std::vector<double> rho_uIp(nDim);
  std::vector<double> GpdxIp(nDim);
  std::vector<double> dpdxIp(nDim);

  // pointers to everyone...
  double *p_uIp = &uIp[0];
  double *p_rho_uIp = &rho_uIp[0];
  double *p_GpdxIp = &GpdxIp[0];
  double *p_dpdxIp = &dpdxIp[0];

  // deal with state
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);

  ScalarFieldType* Udiag = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "momentum_diag");

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)  
    & !(realm_.get_inactive_selector());

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

    // algorithm related
    ws_vrtm.resize(nodesPerElement*nDim);
    ws_Gpdx.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_pressure.resize(nodesPerElement);
    ws_density.resize(nodesPerElement);
    ws_udiag.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_dndx.resize(nDim*numScsIp*nodesPerElement);
    ws_deriv.resize(nDim*numScsIp*nodesPerElement);
    ws_det_j.resize(numScsIp);
    ws_shape_function.resize(numScsIp*nodesPerElement);

    // pointers
    double *p_vrtm = &ws_vrtm[0];
    double *p_Gpdx = &ws_Gpdx[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_pressure = &ws_pressure[0];
    double *p_density = &ws_density[0];
    double *p_udiag = &ws_udiag[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_dndx = &ws_dndx[0];
    double *p_shape_function = &ws_shape_function[0];
    
    if ( shiftMdot_)
      meSCS->shifted_shape_fcn(&p_shape_function[0]);
    else
      meSCS->shape_fcn(&p_shape_function[0]);
    
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // pointers to elem data
      double * mdot = stk::mesh::field_data(*massFlowRate_, b, k );

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        const double * vrtm   = stk::mesh::field_data(*velocityRTM_, node);
        const double * Gjp    = stk::mesh::field_data(*Gpdx_, node);
        const double * coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        p_pressure[ni] = *stk::mesh::field_data(*pressure_, node);
        p_density[ni]  = *stk::mesh::field_data(densityNp1, node);
        p_udiag[ni] = *stk::mesh::field_data(*Udiag, node);

        // gather vectors
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_vrtm[offSet+j] = vrtm[j];
          p_Gpdx[offSet+j] = Gjp[j];
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // compute dndx
      if (shiftPoisson_)
        meSCS->shifted_grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      else
        meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);
      
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // setup for ip values
        for ( int j = 0; j < nDim; ++j ) {
          p_uIp[j] = 0.0;
          p_rho_uIp[j] = 0.0;
          p_GpdxIp[j] = 0.0;
          p_dpdxIp[j] = 0.0;
        }
        double rhoIp = 0.0;
        double projTimeScaleIp = 0.0;

        const int offSet = ip*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const double r = p_shape_function[offSet+ic];
          const double nodalPressure = p_pressure[ic];
          const double nodalRho = p_density[ic];
          const double uDiagInv = 1.0 / p_udiag[ic];

          rhoIp += r*nodalRho;
          projTimeScaleIp += r / p_udiag[ic];

          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_GpdxIp[j] += r*p_Gpdx[nDim*ic+j] * uDiagInv;
            p_uIp[j] += r*p_vrtm[nDim*ic+j];
            p_rho_uIp[j] += r*nodalRho*p_vrtm[nDim*ic+j];
            p_dpdxIp[j] += p_dndx[offSetDnDx+j]*nodalPressure;
          }
        }

        // assemble mdot
        double tmdot = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          tmdot += (interpTogether*p_rho_uIp[j] + om_interpTogether*rhoIp*p_uIp[j] 
                    - (projTimeScaleIp * p_dpdxIp[j] - p_GpdxIp[j]))*p_scs_areav[ip*nDim+j];
        }

        mdot[ip] = tmdot;

      }
    }
  }

  // check for edge-mdot assembly
  if ( assembleMdotToEdge_ )
    assemble_edge_mdot();
}

//--------------------------------------------------------------------------
//-------- assemble_edge_mdot ----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMdotElemAlgorithm::assemble_edge_mdot()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // zero out edge mdot
  stk::mesh::Selector s_all_edges
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);
  stk::mesh::BucketVector const& edge_buckets =
    realm_.get_buckets( stk::topology::EDGE_RANK, s_all_edges );
  for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
        ib != edge_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * edgeMdot = stk::mesh::field_data(*edgeMassFlowRate_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        edgeMdot[k] = 0.0;
    }
  }

  // now assemble by looping over elements; looks like the edge-assembled area
  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
      &stk::mesh::selectUnion(partVec_);
  stk::mesh::BucketVector const& element_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  for ( stk::mesh::BucketVector::const_iterator ib = element_buckets.begin();
        ib != element_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

    // extract master element specifics
    const int *lrscv = meSCS->adjacentNodes();

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // extract element
      stk::mesh::Entity elem = b[k];

      // ip data for this element; scs mdot
      const double *scsMdot = stk::mesh::field_data(*massFlowRate_, elem );

      // Use node Entity because we'll need to call BulkData::identifier(.).
      stk::mesh::Entity const * elem_node_rels = b.begin_nodes(k);

      // iterate edges
      stk::mesh::Entity const * elem_edge_rels = b.begin_edges(k);
      int num_edges = b.num_edges(k);

      for ( int nedge = 0; nedge < num_edges; ++nedge ) {

        // get edge and area_vector
        stk::mesh::Entity edge = elem_edge_rels[nedge];
        double * edgeMdot = stk::mesh::field_data(*edgeMassFlowRate_, edge );

        // extract edge->node relations
        stk::mesh::Entity const * edge_node_rels = bulk_data.begin_nodes(edge);
        ThrowAssert( 2 == bulk_data.num_nodes(edge) );

        // work towards "sign" convention

        // extract a local node; choose to pick L and follow it through
        const int iloc_L = lrscv[2*nedge];

        // get global identifiers for nodes Left and Right from the element
        const size_t iglob_Lelem = bulk_data.identifier(elem_node_rels[iloc_L]);
        const size_t iglob_Ledge = bulk_data.identifier(edge_node_rels[0]);

        // determine the sign value for area vector; if Left node is the same,
        // then the element and edge relations are aligned
        const double sign = ( iglob_Lelem == iglob_Ledge ) ? 1.0 : -1.0;
        *edgeMdot += scsMdot[nedge]*sign;
      }
    }
  }

  // parallel reduce
  stk::mesh::parallel_sum(bulk_data, {edgeMassFlowRate_});

}


} // namespace nalu
} // namespace Sierra
