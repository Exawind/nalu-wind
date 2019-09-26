/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <TurbViscSSTAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbViscSSTAlgorithm - compute tvisc for Smagorinsky model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbViscSSTAlgorithm::TurbViscSSTAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    aOne_(realm.get_turb_model_constant(TM_aOne)),
    betaStar_(realm.get_turb_model_constant(TM_betaStar)),
    density_(NULL),
    viscosity_(NULL),
    tke_(NULL),
    sdr_(NULL),
    minDistance_(NULL),
    dudx_(NULL),
    tvisc_(NULL)
{
  // 2003 variant; basically, sijMag replaces vorticityMag
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  tke_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  sdr_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_dissipation_rate");
  minDistance_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "minimum_distance_to_wall");
  dudx_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbViscSSTAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*tvisc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double *rho = stk::mesh::field_data(*density_, b);
    const double *visc = stk::mesh::field_data(*viscosity_, b);
    const double *tke = stk::mesh::field_data(*tke_, b);
    const double *sdr = stk::mesh::field_data(*sdr_, b);
    const double *minD = stk::mesh::field_data(*minDistance_, b);
    double *tvisc = stk::mesh::field_data(*tvisc_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // compute strain rate magnitude; pull pointer within the loop to make it managable
      const double *dudx = stk::mesh::field_data(*dudx_, b[k] );
      double sijMag = 0.0;
      for ( int i = 0; i < nDim; ++i ) {
        const int offSet = nDim*i;
        for ( int j = 0; j < nDim; ++j ) {
          const double rateOfStrain = 0.5*(dudx[offSet+j] + dudx[nDim*j+i]);
          sijMag += rateOfStrain*rateOfStrain;
        }
      }
      sijMag = std::sqrt(2.0*sijMag);

      // some temps
      const double minDSq = minD[k]*minD[k];
      const double trbDiss = std::sqrt(tke[k])/betaStar_/sdr[k]/minD[k];
      const double lamDiss = 500.0*visc[k]/rho[k]/sdr[k]/minDSq;
      const double fArgTwo = std::max(2.0*trbDiss, lamDiss);
      const double fTwo = std::tanh(fArgTwo*fArgTwo);

      tvisc[k] = aOne_*rho[k]*tke[k]/std::max(aOne_*sdr[k], sijMag*fTwo);

    }
  }
}

} // namespace nalu
} // namespace Sierra
