#include <property_evaluator/HDF5TablePropAlgorithm.h>
#include <tabular_props/HDF5Table.h>
#include <tabular_props/H5IO.h>

#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <NaluEnv.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace sierra {
namespace nalu {

//============================================================================
HDF5TablePropAlgorithm::HDF5TablePropAlgorithm(
  Realm & realm,
  stk::mesh::Part * part,
  H5IO *fileIO,
  stk::mesh::FieldBase * prop,
  std::string tablePropName,
  std::vector<std::string> &indVarNameVec,
  std::vector<std::string> &indVarTableNameVec,
  const stk::mesh::MetaData &meta_data)
  : Algorithm(realm, part),
    prop_(prop),
    tablePropName_(tablePropName),
    indVarTableNameVec_(indVarTableNameVec),
    indVarSize_(indVarNameVec.size()),
    fileIO_( fileIO )
{ 
  // extract the independent fields; check if there is one..
  if ( indVarSize_ == 0 )
    throw std::runtime_error("HDF5TablePropAlgorithm: independent variable size is zero:");

  indVar_.resize(indVarSize_);
  for ( size_t k = 0; k < indVarSize_; ++k) {
    ScalarFieldType *indVar = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, indVarNameVec[k]);
    if ( NULL == indVar ) {
      throw std::runtime_error("HDF5TablePropAlgorithm: independent variable not registered:");
    }
    else {
      indVar_[k] = indVar;
    }
  }
  
  // resize some work vectors
  workIndVar_.resize(indVarSize_);
  workZ_.resize(indVarSize_);

  //read in table
  //read_hdf5( );
  table_ = new HDF5Table( fileIO, tablePropName_, indVarNameVec, indVarTableNameVec ) ;

  // provide some output
  NaluEnv::self().naluOutputP0() << "the Following Table Property name will be extracted: " << tablePropName << std::endl;
  for ( size_t k = 0; k < indVarTableNameVec_.size(); ++k ) {
    NaluEnv::self().naluOutputP0() << "using independent variables: " << indVarTableNameVec_[k] << std::endl;
  }
}
//----------------------------------------------------------------------------
HDF5TablePropAlgorithm::~HDF5TablePropAlgorithm()
{
  delete table_;
}
//----------------------------------------------------------------------------
void
HDF5TablePropAlgorithm::execute()
{
  // make sure that partVec_ is size one
  ThrowAssert( partVec_.size() == 1 );

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, selector );

  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // property is always single in size
    double *prop  = (double*) stk::mesh::field_data(*prop_, b);

    // independent variable size can be more than one
    for ( size_t l = 0; l < indVarSize_; ++l) {
      double *indVar  = (double*) stk::mesh::field_data(*indVar_[l], b);
      workIndVar_[l] = indVar;
    }

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // load up the independent work array
      for ( size_t l = 0; l < indVarSize_; ++l) {
        double *z = workIndVar_[l];
        workZ_[l] = z[k];
      }
      prop[k] = table_->query( workZ_ );
    }
  }
}
//============================================================================

} // end nalu namespace
} // end sierra namespace

