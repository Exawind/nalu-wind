/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <PstabErrorIndicatorElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
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
// PstabErrorIndicatorElemAlgorithm - error indicator for element fluids
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
PstabErrorIndicatorElemAlgorithm::PstabErrorIndicatorElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *pressure,
  VectorFieldType *Gpdx,
  const bool simpleGradApproach)
  : Algorithm(realm, part),
    pressure_(pressure),
    Gpdx_(Gpdx),
    coordinates_(NULL),
    pstabEI_(NULL),
    simpleGradApproachScale_(simpleGradApproach ? 0.0 : 1.0)
{
   // extract fields; nodal
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  pstabEI_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
PstabErrorIndicatorElemAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // time step
  const double dt = realm_.get_time_step();
  const double gamma1 = realm_.get_gamma1();
  const double projTimeScale = dt/gamma1;

  // nodal fields to gather
  std::vector<double> ws_Gpdx;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_pressure;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_dndx;
  std::vector<double> ws_deriv;
  std::vector<double> ws_det_j;
  std::vector<double> ws_shape_function;

  // integration point data that depends on size
  std::vector<double> GpdxIp(nDim);
  std::vector<double> dpdxIp(nDim);

  // pointers to everyone...
  double *p_GpdxIp = &GpdxIp[0];
  double *p_dpdxIp = &dpdxIp[0];

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
    const int numScsIp = meSCS->num_integration_points();

    // algorithm related
    ws_Gpdx.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_pressure.resize(nodesPerElement);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_dndx.resize(nDim*numScsIp*nodesPerElement);
    ws_deriv.resize(nDim*numScsIp*nodesPerElement);
    ws_det_j.resize(numScsIp);
    ws_shape_function.resize(numScsIp*nodesPerElement);

    // pointers
    double *p_Gpdx = &ws_Gpdx[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_pressure = &ws_pressure[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_dndx = &ws_dndx[0];
    double *p_shape_function = &ws_shape_function[0];

    meSCS->shape_fcn(&p_shape_function[0]);

    // pointers to elem data
    double * pstabEI = stk::mesh::field_data(*pstabEI_, b );

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

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
        double * Gjp    = stk::mesh::field_data(*Gpdx_, node);
        double * coords = stk::mesh::field_data(*coordinates_, node);

        // gather scalars
        p_pressure[ni] = *stk::mesh::field_data(*pressure_, node);

        // gather vectors
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_Gpdx[offSet+j] = Gjp[j];
          p_coordinates[offSet+j] = coords[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // compute dndx
      meSCS->grad_op(1, &p_coordinates[0], &p_dndx[0], &ws_deriv[0], &ws_det_j[0], &scs_error);

      // initialize error indicator
      double errorIndicator = 0.0;

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // setup for ip values
        for ( int j = 0; j < nDim; ++j ) {
          p_GpdxIp[j] = 0.0;
          p_dpdxIp[j] = 0.0;
        }

        const int offSet = ip*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {

          const double r = p_shape_function[offSet+ic];
          const double nodalPressure = p_pressure[ic];

          const int offSetDnDx = nDim*nodesPerElement*ip + ic*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            p_GpdxIp[j] += r*p_Gpdx[nDim*ic+j];
            p_dpdxIp[j] += p_dndx[offSetDnDx+j]*nodalPressure;
          }
        }

        for ( int j = 0; j < nDim; ++j ) {
	  const double theEI = -projTimeScale*(p_dpdxIp[j] - p_GpdxIp[j]*simpleGradApproachScale_)*p_scs_areav[ip*nDim+j];
	  errorIndicator += theEI*theEI;
        }
      }
    
      // scatter to error indicator
      pstabEI[k] = std::sqrt(errorIndicator);
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
PstabErrorIndicatorElemAlgorithm::~PstabErrorIndicatorElemAlgorithm()
{
  // does nothing
}



} // namespace nalu
} // namespace Sierra
