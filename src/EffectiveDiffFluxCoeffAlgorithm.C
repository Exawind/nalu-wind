// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <EffectiveDiffFluxCoeffAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// EffectiveDiffFluxCoeffAlgorithm - compute effective diff flux coeff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EffectiveDiffFluxCoeffAlgorithm::EffectiveDiffFluxCoeffAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* visc,
  ScalarFieldType* tvisc,
  ScalarFieldType* evisc,
  const double sigmaLam,
  const double sigmaTurb)
  : Algorithm(realm, part),
    visc_(visc),
    tvisc_(tvisc),
    evisc_(evisc),
    sigmaLam_(sigmaLam),
    sigmaTurb_(sigmaTurb),
    isTurbulent_(realm_.is_turbulent())
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
EffectiveDiffFluxCoeffAlgorithm::execute()
{

  const double invSigmaLam = 1.0 / sigmaLam_;
  const double invSigmaTurb = 1.0 / sigmaTurb_;

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*visc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);

  if (isTurbulent_) {
    for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
         ib != node_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;
      const stk::mesh::Bucket::size_type length = b.size();

      const double* visc = stk::mesh::field_data(*visc_, b);
      const double* tvisc = stk::mesh::field_data(*tvisc_, b);
      double* evisc = stk::mesh::field_data(*evisc_, b);

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        evisc[k] = visc[k] * invSigmaLam + tvisc[k] * invSigmaTurb;
      }
    }
  } else {
    for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
         ib != node_buckets.end(); ++ib) {
      stk::mesh::Bucket& b = **ib;
      const stk::mesh::Bucket::size_type length = b.size();

      const double* visc = stk::mesh::field_data(*visc_, b);
      double* evisc = stk::mesh::field_data(*evisc_, b);

      for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
        evisc[k] = visc[k] * invSigmaLam;
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
