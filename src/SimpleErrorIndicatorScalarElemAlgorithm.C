/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <SolutionOptions.h>
#include <SimpleErrorIndicatorScalarElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <NaluEnv.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// SimpleErrorIndicatorElemAlgorithm - error indicator for element fluids
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SimpleErrorIndicatorScalarElemAlgorithm::SimpleErrorIndicatorScalarElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx
  )
  : Algorithm(realm, part)
{
   // extract fields; nodal
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  velocity_ = scalarQ;
  dudx_ = dqdx;
  nUnk_ = 1;

  errorIndicatorField_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
SimpleErrorIndicatorScalarElemAlgorithm::execute()
{
  ErrorIndicatorType option = realm_.solutionOptions_->errorIndicatorType_;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_coordinates;
  std::vector<double> ws_dudx;

  // deal with state
  ScalarFieldType& velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  double maxV = 0.0;

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( std::vector<stk::mesh::Bucket*>::const_iterator ib = elem_buckets.begin();
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
    ws_velocityNp1.resize(nodesPerElement*nUnk_);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_dudx.resize(nodesPerElement*nDim*nUnk_);

    // pointers
    double *p_velocityNp1 = &ws_velocityNp1[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_dudx = &ws_dudx[0];

    // pointers to elem data
    double * errorIndicatorField = stk::mesh::field_data(*errorIndicatorField_, b );

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
        const int niNUnk = ni*nUnk_;

        // row for p_dudx
        const int row_p_dudx = niNUnk*nDim;
        for ( int i=0; i < nDim; ++i ) {
          p_coordinates[niNdim+i] = coords[i];
        }
        for ( int i=0; i < nUnk_; ++i ) {
          p_velocityNp1[niNUnk+i] = uNp1[i];
          maxV = std::max(std::fabs(uNp1[i]), maxV);
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
        const int ilNUnk = il*nUnk_;
        const int irNUnk = ir*nUnk_;

        const int ilNdim = il*nDim;
        const int irNdim = ir*nDim;

        const int row_p_dudxL = ilNUnk*nDim;
        const int row_p_dudxR = irNUnk*nDim;

        double dlen = 0.0;
        for ( int i = 0; i < nDim; ++i ) {

          const double dxi = p_coordinates[irNdim+i] - p_coordinates[ilNdim+i];
          dlen += dxi*dxi;
        }
        dlen = std::sqrt(dlen);

        for ( int i = 0; i < nUnk_; ++i ) {
          double duL = 0.0;
          double duR = 0.0;
          double du2 = 0.0;
          for(int j = 0; j < nDim; ++j ) {
            const double dxj = p_coordinates[irNdim+j] - p_coordinates[ilNdim+j];
            duL += dxj*p_dudx[row_p_dudxL+i*nUnk_+j];
            duR += dxj*p_dudx[row_p_dudxR+i*nUnk_+j];
            //du2 += dxj*(p_dudx[row_p_dudxR+i*nUnk_+j] - p_dudx[row_p_dudxL+i*nUnk_+j]);
            du2 += dxj*(p_dudx[row_p_dudxR+i*nUnk_+j] + p_dudx[row_p_dudxL+i*nUnk_+j])/2.0;
          }

          if (option == EIT_SIMPLE_DUDX2) errorIndicator += du2*du2;
        }

#define DUDX(i,j) p_dudx[row_p_dudxL+i*nUnk_+j]
#define X 0
#define Y 1
#define Z 2

        if (option == EIT_SIMPLE_VORTICITY || option == EIT_SIMPLE_VORTICITY_DX)
          {
            double vorticity[3] = {DUDX(Z,Y) - DUDX(Y,Z), DUDX(Z,X) - DUDX(X,Z), nDim == 3 ? (DUDX(Y,X) - DUDX(X,Y)):0};
            if (option == EIT_SIMPLE_VORTICITY) errorIndicator += std::sqrt(vorticity[0]*vorticity[0]+vorticity[1]*vorticity[1]+vorticity[2]*vorticity[2]);
            if (option == EIT_SIMPLE_VORTICITY_DX) errorIndicator += dlen*std::sqrt(vorticity[0]*vorticity[0]+vorticity[1]*vorticity[1]+vorticity[2]*vorticity[2]);
          }
      }

      // scatter to error indicator
      const double totalPts = (double)nUnk_*numScsIp;
      errorIndicator /= totalPts;
      if (option == EIT_SIMPLE_DUDX2) {
        errorIndicator = std::sqrt(errorIndicator);
        //errorIndicator = std::sqrt(errorIndicator);
      }
      if (option == EIT_SIMPLE_VORTICITY || option == EIT_SIMPLE_VORTICITY_DX)
        errorIndicator = std::sqrt(errorIndicator);

      errorIndicatorField[k] = errorIndicator;
      if (errorIndicatorField[k] < 0.0)
        throw std::logic_error("ERROR in SimpleErrorIndicatorElemAlgorithm: limiter value exceeds unity.");
    }
  }

  stk::ParallelMachine pm = realm_.bulk_data().parallel();
  stk::all_reduce( pm, stk::ReduceMax<1>( &maxV ) );
  NaluEnv::self().naluOutputP0() << "tmp srk maxV= " << maxV << std::endl;

}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SimpleErrorIndicatorScalarElemAlgorithm::~SimpleErrorIndicatorScalarElemAlgorithm()
{
  // does nothing
}


} // namespace nalu
} // namespace Sierra
