// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <property_evaluator/ThermalConductivityFromPrandtlPropAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ThermalConductivityFromPrandtlPropAlgorithm - compute k from Pr
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ThermalConductivityFromPrandtlPropAlgorithm::
  ThermalConductivityFromPrandtlPropAlgorithm(
    Realm& realm,
    const stk::mesh::PartVector& part_vec,
    ScalarFieldType* thermalCond,
    ScalarFieldType* specHeat,
    ScalarFieldType* viscosity,
    const double Pr)
  : Algorithm(realm, part_vec),
    thermalCond_(thermalCond),
    specHeat_(specHeat),
    viscosity_(viscosity),
    Pr_(Pr)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ThermalConductivityFromPrandtlPropAlgorithm::execute()
{
  // make sure that partVec_ is size one
  STK_ThrowAssert(partVec_.size() == 1);

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  thermalCond_->sync_to_host();
  specHeat_->sync_to_host();
  viscosity_->sync_to_host();

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* thermalCond = stk::mesh::field_data(*thermalCond_, b);
    const double* specHeat = stk::mesh::field_data(*specHeat_, b);
    const double* viscosity = stk::mesh::field_data(*viscosity_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      thermalCond[k] = specHeat[k] * viscosity[k] / Pr_;
    }
  }
  thermalCond_->modify_on_host();
  thermalCond_->sync_to_device();
}

} // namespace nalu
} // namespace sierra
