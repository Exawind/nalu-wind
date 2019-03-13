/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <EffectiveDESABLDiffFluxCoeffAlgorithm.h>
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
// EffectiveDESABLDiffFluxCoeffAlgorithm - compute effective diff flux coeff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EffectiveDESABLDiffFluxCoeffAlgorithm::EffectiveSSTDiffFluxCoeffAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *visc,
  ScalarFieldType *tvisc,
  ScalarFieldType *evisc,
  const double sigmaLam,
  const double sigmaTurb,
  const double sigmaOne,
  const double sigmaTwo)
  : Algorithm(realm, part),
    visc_(visc),
    tvisc_(tvisc),
    evisc_(evisc),
    fOneBlend_(NULL),
    fABLBlending_(NULL),	
    sigmaLam_(sigmaLam),
    sigmaTurb_(sigmaTurb),	
    sigmaOne_(sigmaOne),
    sigmaTwo_(sigmaTwo)
{
  // extract blending nodal variable
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  fOneBlend_= meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sst_f_one_blending");
  fABLBlending_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK,"abl_des_f_blending");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
EffectiveDESABLTDiffFluxCoeffAlgorithm::execute()
{

  const double invSigmaLam = 1.0/sigmaLam_;
  const double invSigmaTurb = 1.0/sigmaTurb_;

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // define some common selectors
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*visc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );

  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
      ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    const double * visc = stk::mesh::field_data(*visc_, b);
    const double * tvisc = stk::mesh::field_data(*tvisc_, b);
    const double * fOneBlend = stk::mesh::field_data(*fOneBlend_, b);
    const double *fABLBlend = stk::mesh::field_data(*fABLBlending_, b);
    double * evisc = stk::mesh::field_data(*evisc_, b);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double blendedConstant = fOneBlend[k]*sigmaOne_ + (1.0-fOneBlend[k])*sigmaTwo_;
      
      const double blendedConstantDESABL = fABLBlend[k]*invSigmaTurb + (1-fABLBlend[k])*blendedConstant;              
  

      evisc[k] = visc[k]*invSigmaLam + tvisc[k]*blendedConstantDESABL;
    }
  }
}

} // namespace nalu
} // namespace Sierra
