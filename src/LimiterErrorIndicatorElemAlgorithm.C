/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <LimiterErrorIndicatorElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
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
// LimiterErrorIndicatorElemAlgorithm - error indicator for element fluids
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
LimiterErrorIndicatorElemAlgorithm::LimiterErrorIndicatorElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part)
{
   // extract fields; nodal
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  LimiterEI_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
LimiterErrorIndicatorElemAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double small = 1.0e-16;

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_dudx;

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

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

    // algorithm related
    ws_velocityNp1.resize(nodesPerElement*nDim);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_dudx.resize(nodesPerElement*nDim*nDim);

    // pointers
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_dudx = &ws_dudx[0];

    // pointers to elem data
    double * LimiterEI = stk::mesh::field_data(*LimiterEI_, b );

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
        const double * uNp1   =  stk::mesh::field_data(velocityNp1, node);
        const double * coords =  stk::mesh::field_data(*coordinates_, node);
        const double * du     =  stk::mesh::field_data(*dudx_, node);

        // gather vectors
        const int niNdim = ni*nDim;

        // row for p_dudx
        const int row_p_dudx = niNdim*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_velocityNp1[niNdim+i] = uNp1[i];
          p_coordinates[niNdim+i] = coords[i];
          // gather tensor
          const int row_dudx = i*nDim;
          for ( int j=0; j < nDim; ++j ) {
            p_dudx[row_p_dudx+row_dudx+j] = du[row_dudx+j];
          }
        }
      }

      // initialize error indicator
      double errorIndicator = 0.0;

      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        // save off some offsets
        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        const int row_p_dudxL = ilNdim*nDim;
        const int row_p_dudxR = irNdim*nDim;

        double dlen = 0.0;
        for ( int i = 0; i < nDim; ++i ) {

          const double dxi = p_coordinates[irNdim+i] - p_coordinates[ilNdim+i];
          dlen += dxi*dxi;

          double duL = 0.0;
          double duR = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxj = p_coordinates[irNdim+j] - p_coordinates[ilNdim+j];
            duL += dxj*p_dudx[row_p_dudxL+i*nDim+j];
            duR += dxj*p_dudx[row_p_dudxR+i*nDim+j];
          }
          dlen = std::sqrt(dlen);

          const double du = p_velocityNp1[irNdim+i] - p_velocityNp1[ilNdim+i];
          const double dqMl = 2.0*2.0*duL - du;
          const double dqMr = 2.0*2.0*duR - du;
          const double limitL = van_leer(dqMl, du, small);
          const double limitR = van_leer(dqMr, du, small);

          // FIXME: consider alternative: min(limitL, limitR)
          errorIndicator += std::sqrt(0.5*(limitL*limitL + limitR*limitR));
        }
      }

      // scatter to error indicator
      const double totalPts = (double)nDim*numScsIp;
      const double eps = 1.e-8;
      errorIndicator /= totalPts;
      LimiterEI[k] = (1.0+eps) - errorIndicator;
      if (LimiterEI[k] < 0.0)
        throw std::logic_error("ERROR in LimiterErrorIndicatorElemAlgorithm: limiter value exceeds unity.");
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
LimiterErrorIndicatorElemAlgorithm::~LimiterErrorIndicatorElemAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- van_leer ---------------------------------------------------------
//--------------------------------------------------------------------------
double
LimiterErrorIndicatorElemAlgorithm::van_leer(
  const double &dqm,
  const double &dqp,
  const double &small)
{
  double limit = (2.0*(dqm*dqp+std::abs(dqm*dqp))) /
    ((dqm+dqp)*(dqm+dqp)+small);
  return limit;
}

} // namespace nalu
} // namespace Sierra
