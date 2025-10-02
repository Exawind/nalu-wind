// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <TurbViscSmagorinskyAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// TurbViscSmagorinskyAlgorithm - compute tvisc for Smagorinsky model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbViscSmagorinskyAlgorithm::TurbViscSmagorinskyAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    dudx_(NULL),
    density_(NULL),
    tvisc_(NULL),
    dualNodalVolume_(NULL),
    cmuCs_(realm.get_turb_model_constant(TM_cmuCs))
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  dudx_ = meta_data.get_field<double>(stk::topology::NODE_RANK, "dudx");
  density_ = meta_data.get_field<double>(stk::topology::NODE_RANK, "density");
  tvisc_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbViscSmagorinskyAlgorithm::execute()
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  const double cmuCs = cmuCs_;
  const double invNdim = 1.0 / meta_data.spatial_dimension();

  // define some common selectors
  stk::mesh::Selector s_all_nodes =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*tvisc_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, s_all_nodes);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    const double* density = stk::mesh::field_data(*density_, b);
    const double* dualNodalVolume = stk::mesh::field_data(*dualNodalVolume_, b);
    double* tvisc = stk::mesh::field_data(*tvisc_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      const double* dudx = stk::mesh::field_data(*dudx_, b[k]);

      double sijMag = 0.0;
      for (int i = 0; i < nDim; ++i) {
        const int offSet = nDim * i;
        for (int j = 0; j < nDim; ++j) {
          const double rateOfStrain =
            0.5 * (dudx[offSet + j] + dudx[nDim * j + i]);
          sijMag += rateOfStrain * rateOfStrain;
        }
      }
      sijMag = std::sqrt(2.0 * sijMag);

      const double filter = std::pow(dualNodalVolume[k], invNdim);
      tvisc[k] = cmuCs * cmuCs * density[k] * filter * filter * sijMag;
    }
  }
}

} // namespace nalu
} // namespace sierra
