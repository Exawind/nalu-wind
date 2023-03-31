// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <TurbViscWaleAlgorithm.h>
#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// TurbViscWaleAlgorithm - compute tvisc for Ksgs model
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbViscWaleAlgorithm::TurbViscWaleAlgorithm(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    dudx_(NULL),
    density_(NULL),
    tvisc_(NULL),
    dualNodalVolume_(NULL),
    Cw_(realm.get_turb_model_constant(TM_Cw)),
    kappa_(realm.get_turb_model_constant(TM_kappa))
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  dudx_ =
    meta_data.get_field<TensorFieldType>(stk::topology::NODE_RANK, "dudx");
  density_ =
    meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  tvisc_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "turbulent_viscosity");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  // need NDTW...
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbViscWaleAlgorithm::execute()
{

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  const double invNdim = 1.0 / double(nDim);
  std::vector<double> dudxSq(nDim * nDim, 0);

  // save some factors
  const double threeHalves = 3.0 / 2.0;
  const double fiveHalves = 5.0 / 2.0;
  const double fiveFourths = 5.0 / 4.0;
  const double small = 1.0e-8;

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

      for (int i = 0; i < nDim; ++i) {
        for (int j = 0; j < nDim; ++j) {
          double acc = 0;
          for (int l = 0; l < nDim; ++l) {
            acc += dudx[i * nDim + l] * dudx[l * nDim + j];
          }
          dudxSq[i * nDim + j] = acc;
        }
      }

      double traceDudxSq = 0;
      for (int i = 0; i < nDim; ++i) {
        traceDudxSq += dudxSq[i * nDim + i];
      }

      double SijSq = 0.0;
      double SijdSq = 0.0;
      for (int i = 0; i < nDim; ++i) {
        const int offSetI = nDim * i;
        for (int j = 0; j < nDim; ++j) {
          const int offSetJ = nDim * j;
          const double Sij = 0.5 * (dudx[offSetI + j] + dudx[offSetJ + i]);
          const double traceKron = (i == j) ? traceDudxSq / nDim : 0;
          const double Sijd =
            0.5 * (dudxSq[offSetI + j] + dudxSq[offSetJ + i]) - traceKron;
          SijSq += Sij * Sij;
          SijdSq += Sijd * Sijd;
        }
      }

      const double filter = std::pow(dualNodalVolume[k], invNdim);
      const double Ls = Cw_ * filter; // min (Cw_*filter, kappa*ndtw)
      const double numer = std::pow(SijdSq, threeHalves) + small * small;
      const double demom =
        std::pow(SijSq, fiveHalves) + std::pow(SijdSq, fiveFourths) + small;
      tvisc[k] = density[k] * Ls * Ls * numer / demom;
    }
  }
}

} // namespace nalu
} // namespace sierra
