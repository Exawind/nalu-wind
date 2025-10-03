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
#include <FieldManager.h>
#include <FieldTypeDef.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

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
    Realm& realm, const stk::mesh::PartVector& part_vec, const double Pr)
  : Algorithm(realm, part_vec), Pr_(Pr)
{
  fieldManager_.register_field<double>("thermal_conductivity", part_vec);
  fieldManager_.register_field<double>("specific_heat", part_vec);
  fieldManager_.register_field<double>("viscosity", part_vec);
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

  auto thermalCond =
    fieldManager_.get_legacy_smart_field<double, tags::READ_WRITE>(
      "thermal_conductivity");
  const auto specHeat =
    fieldManager_.get_legacy_smart_field<double, tags::READ>("specific_heat");
  const auto viscosity =
    fieldManager_.get_legacy_smart_field<double, tags::READ>("viscosity");

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      thermalCond(b)[k] = specHeat(b)[k] * viscosity(b)[k] / Pr_;
    }
  }
}

} // namespace nalu
} // namespace sierra
