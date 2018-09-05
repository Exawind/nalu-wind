/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <AssembleWallHeatTransferAlgorithmDriver.h>
#include <Algorithm.h>
#include <AlgorithmDriver.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// AssembleWallHeatTransferAlgorithmDriver - Drives nodal grad algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleWallHeatTransferAlgorithmDriver::AssembleWallHeatTransferAlgorithmDriver(
  Realm &realm)
  : AlgorithmDriver(realm),
    assembledWallArea_(NULL),
    referenceTemperature_(NULL),
    heatTransferCoefficient_(NULL),
    normalHeatFlux_(NULL),
    robinCouplingParameter_(NULL),
    temperature_(NULL)
{
  // register the fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_ht");
  referenceTemperature_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "reference_temperature");
  heatTransferCoefficient_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "heat_transfer_coefficient");
  normalHeatFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "normal_heat_flux");
  robinCouplingParameter_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "robin_coupling_parameter");
  temperature_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AssembleWallHeatTransferAlgorithmDriver::~AssembleWallHeatTransferAlgorithmDriver()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleWallHeatTransferAlgorithmDriver::pre_work()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors; select all nodes (locally and shared)
  // where assembledWallArea is defined
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*assembledWallArea_);

  //===========================================================
  // zero out nodal fields
  //===========================================================

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();
    double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * referenceTemperature = stk::mesh::field_data(*referenceTemperature_, b);
    double * heatTransferCoefficient = stk::mesh::field_data(*heatTransferCoefficient_, b);
    double * normalHeatFlux = stk::mesh::field_data(*normalHeatFlux_, b);
    double * robinCouplingParameter = stk::mesh::field_data(*robinCouplingParameter_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      assembledWallArea[k] = 0.0;
      referenceTemperature[k] = 0.0;
      heatTransferCoefficient[k] = 0.0;
      normalHeatFlux[k] = 0.0;
      robinCouplingParameter[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleWallHeatTransferAlgorithmDriver::post_work()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  std::vector<const stk::mesh::FieldBase*> fields = {
    assembledWallArea_,
    referenceTemperature_,
    heatTransferCoefficient_,
    normalHeatFlux_,
    robinCouplingParameter_
  };
  const std::vector<const stk::mesh::FieldBase*>& const_fields_ref = fields;

  stk::mesh::parallel_sum(bulk_data, const_fields_ref);

  // add periodic assembly piror to normalization
  if ( realm_.hasPeriodic_) {
    const unsigned scalarSize = 1;
    const bool bypassFieldCheck = false; // nodal fields are only defined at periodic nodes
    realm_.periodic_field_update(assembledWallArea_, scalarSize, bypassFieldCheck);
    realm_.periodic_field_update(referenceTemperature_, scalarSize, bypassFieldCheck);
    realm_.periodic_field_update(heatTransferCoefficient_, scalarSize, bypassFieldCheck);
    realm_.periodic_field_update(normalHeatFlux_, scalarSize, bypassFieldCheck);
    realm_.periodic_field_update(robinCouplingParameter_, scalarSize, bypassFieldCheck);
  }

  // normalize
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*assembledWallArea_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();
    const double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    const double * temperature = stk::mesh::field_data(*temperature_, b);
    double * referenceTemperature = stk::mesh::field_data(*referenceTemperature_, b);
    double * heatTransferCoefficient = stk::mesh::field_data(*heatTransferCoefficient_, b);
    double * normalHeatFlux = stk::mesh::field_data(*normalHeatFlux_, b);
    double * robinCouplingParameter = stk::mesh::field_data(*robinCouplingParameter_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double ak = assembledWallArea[k];
      const double htc = -heatTransferCoefficient[k]/ak/temperature[k];
      heatTransferCoefficient[k] = htc;
      referenceTemperature[k] /= (ak*htc);
      normalHeatFlux[k] /= ak;
      robinCouplingParameter[k] /= ak;
    }
  }

}

} // namespace nalu
} // namespace Sierra
