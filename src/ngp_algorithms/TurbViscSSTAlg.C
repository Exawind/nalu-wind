// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/TurbViscSSTAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"


namespace sierra {
namespace nalu {

TurbViscSSTAlg::TurbViscSSTAlg(
  Realm& realm,
  stk::mesh::Part* part,
  ScalarFieldType* tvisc,
  const bool useAverages)
  : Algorithm(realm, part),
    tviscField_(tvisc),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    tke_(get_field_ordinal(realm.meta_data(), "turbulent_ke")),
    sdr_(get_field_ordinal(realm.meta_data(), "specific_dissipation_rate")),
    minDistance_(
      get_field_ordinal(realm.meta_data(), "minimum_distance_to_wall")),
    dwalldistdx_(
      get_field_ordinal(realm.meta_data(), "dwalldistdx")),
    dnDotVdx_(
      get_field_ordinal(realm.meta_data(), "dnDotVdx")),
    dudx_(get_field_ordinal(
      realm.meta_data(), (useAverages) ? "average_dudx" : "dudx")),
    velocity_(get_field_ordinal(realm.meta_data(), "velocity")),
    dpdx_(get_field_ordinal(realm.meta_data(), "dpdx")),
    tvisc_(tvisc->mesh_meta_data_ordinal()),
    aOne_(realm.get_turb_model_constant(TM_aOne)),
    sThres_(realm.get_turb_model_constant(TM_sThres)),
    betaStar_(realm.get_turb_model_constant(TM_betaStar))
{
}

void
TurbViscSSTAlg::execute()
{
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();

  stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*tviscField_);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto density = fieldMgr.get_field<double>(density_);
  const auto visc = fieldMgr.get_field<double>(viscosity_);
  const auto tke = fieldMgr.get_field<double>(tke_);
  const auto sdr = fieldMgr.get_field<double>(sdr_);
  const auto minD = fieldMgr.get_field<double>(minDistance_);
  const auto dwalldistdx = fieldMgr.get_field<double>(dwalldistdx_);
  const auto dnDotVdx = fieldMgr.get_field<double>(dnDotVdx_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  const auto velocity = fieldMgr.get_field<double>(velocity_);
  const auto dpdx = fieldMgr.get_field<double>(dpdx_);
  auto tvisc = fieldMgr.get_field<double>(tvisc_);

  tvisc.sync_to_device();

  const DblType aOne = aOne_;
  const DblType sThres = sThres_;
  const DblType betaStar = betaStar_;
  const int nDim = meta.spatial_dimension();

  nalu_ngp::run_entity_algorithm(
    "TurbViscSSTAlg", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& meshIdx) {
      DblType sijMag = 0.0;
      for (int i = 0; i < nDim; ++i) {
        const int offSet = nDim * i;
        for (int j = 0; j < nDim; ++j) {
          const DblType rateOfStrain = 0.5 * (dudx.get(meshIdx, offSet + j) +
                                              dudx.get(meshIdx, nDim * j + i));
          sijMag += rateOfStrain * rateOfStrain;
        }
      }
      sijMag = stk::math::sqrt(2.0 * sijMag);

      // Compute udpdx
      DblType u_mag = 0.0;
      for (int i = 0; i < nDim; ++i) {
	u_mag += velocity.get(meshIdx, i) * velocity.get(meshIdx, i);
      }
      u_mag = stk::math::sqrt(u_mag);
      if (u_mag < 1e-8)
	u_mag = 1e-8;
      DblType udpdx = 0.0;
      for (int i = 0; i < nDim; ++i) {
	udpdx += velocity.get(meshIdx, i)/u_mag * dpdx.get(meshIdx, i);
      }
      
      const DblType minDSq = minD.get(meshIdx, 0) * minD.get(meshIdx, 0);
      const DblType trbDiss = stk::math::sqrt(tke.get(meshIdx, 0)) / betaStar /
                              sdr.get(meshIdx, 0) / minD.get(meshIdx, 0);
      const DblType lamDiss = 500.0 * visc.get(meshIdx, 0) /
                              density.get(meshIdx, 0) / sdr.get(meshIdx, 0) /
                              minDSq;
      const DblType fArgTwo = stk::math::max(2.0 * trbDiss, lamDiss);
      const DblType fTwo = stk::math::tanh(fArgTwo * fArgTwo);

      // master
      //tvisc.get(meshIdx, 0) =
      //  aOne * density.get(meshIdx, 0) * tke.get(meshIdx, 0) /
      //  stk::math::max(aOne * sdr.get(meshIdx, 0), sijMag * fTwo);

      DblType dvnn = 0.0;
      DblType lamda0L = 0.0;

      for (int i = 0; i < nDim; ++i) {
	dvnn += dwalldistdx.get(meshIdx, i) * dnDotVdx.get(meshIdx, i);
      }
      lamda0L = -7.57e-3 * dvnn * minD.get(meshIdx, 0) * minD.get(meshIdx, 0) * density.get(meshIdx, 0) / visc.get(meshIdx, 0) + 0.0128;
      lamda0L = stk::math::min(stk::math::max(lamda0L, -1.0), 1.0);

      // udpdx sensor, but sijMag eddy viscosity
      if (0.31 * sdr.get(meshIdx, 0) > sijMag * fTwo){ // non-APG model
	tvisc.get(meshIdx, 0) =
	  density.get(meshIdx, 0) * tke.get(meshIdx, 0) /
	  (sdr.get(meshIdx, 0));
      } else if (lamda0L<-0.0681) { // laminar separation criterion satisfied (this is functioning as the APG sensor)
	tvisc.get(meshIdx, 0) =
	  aOne * density.get(meshIdx, 0) * tke.get(meshIdx, 0) /
	  (sijMag * fTwo);
      } else { // pseudo-APG model, used in corner cases
	tvisc.get(meshIdx, 0) =
	  0.31 * density.get(meshIdx, 0) * tke.get(meshIdx, 0) /
	  (sijMag * fTwo);
      }
      
    });
  tvisc.modify_on_device();
}

} // namespace nalu
} // namespace sierra
