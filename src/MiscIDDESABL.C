// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <MiscIDDESABL.h>
#include <Realm.h>
#include <cmath>
#include <iostream>
#include <array>
#include "stk_mesh/base/MetaData.hpp"
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

void
register_iddes_abl_fields(stk::mesh::MetaData& meta_data, stk::mesh::Part* part)
{
  const int nDim = meta_data.spatial_dimension();
  VectorFieldType& velocity_abl = meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity_abl");
  VectorFieldType& velocity_rans = meta_data.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity_rans");
  ScalarFieldType& tke_rans = meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "tke_rans");
  ScalarFieldType& tke_abl = meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "tke_abl");
  ScalarFieldType& sdr_rans = meta_data.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "sdr_rans");
  stk::mesh::put_field_on_mesh(velocity_abl, *part, nDim, nullptr);
  stk::mesh::put_field_on_mesh(velocity_rans, *part, nDim, nullptr);
  stk::mesh::put_field_on_mesh(tke_abl, *part, nullptr);
  stk::mesh::put_field_on_mesh(tke_rans, *part, nullptr);
  stk::mesh::put_field_on_mesh(sdr_rans, *part, nullptr);
}

void
initial_work_iddes_abl(Realm& realm)
{
  if (realm.get_current_time() == 0.0) {
    auto& meta = realm.meta_data();
    auto& bulk = realm.bulk_data();
    auto* velocity =
      meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    auto* ndtw = meta.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "minimum_distance_to_wall");
    auto* dual_nodal_volume = meta.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume");
    auto* velocity_abl =
      meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_abl");
    auto* velocity_rans = meta.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "velocity_rans");
    auto* tke_abl =
      meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_abl");
    auto* tke_rans =
      meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_rans");
    auto* sdr_rans =
      meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "sdr_rans");
    auto* turbKinEnergy =
      meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
    auto* specDissRate = meta.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "specific_dissipation_rate");

    const double b_ndtw = realm.get_turb_model_constant(TM_abl_bndtw);
    const double delta_ndtw = realm.get_turb_model_constant(TM_abl_deltandtw);
    std::array<double, 3> hubVelVec{0.0, 0.0, 0.0};
    hubVelVec[0] = 8.0 * std::cos(40.0 * M_PI / 180.0);
    hubVelVec[1] = 8.0 * std::sin(40.0 * M_PI / 180.0);
    hubVelVec[2] = 0.0;

    const stk::mesh::BucketVector& node_buckets =
      bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part());

    for (auto b : node_buckets) {
      for (size_t in = 0; in < b->size(); in++) {
        auto node = (*b)[in];
        double dnv = *(stk::mesh::field_data(*dual_nodal_volume, node));
        double minD = *(stk::mesh::field_data(*ndtw, node));
        double* vel_abl = stk::mesh::field_data(*velocity_abl, node);
        double* vel_rans = stk::mesh::field_data(*velocity_rans, node);
        double* vel = stk::mesh::field_data(*velocity, node);
        double tkeabl = *(stk::mesh::field_data(*tke_abl, node));
        double tkerans = *(stk::mesh::field_data(*tke_rans, node));
        double* tke = stk::mesh::field_data(*turbKinEnergy, node);
        double sdrrans = *(stk::mesh::field_data(*sdr_rans, node));
        double* sdr = stk::mesh::field_data(*specDissRate, node);

        double f_des_abl = 0.5 * std::tanh((b_ndtw - minD) / delta_ndtw) + 0.5;

        for (auto i = 0; i < 3; i++)
          vel[i] =
            f_des_abl * vel_rans[i] +
            (1.0 - f_des_abl) * (vel_abl[i] + vel_rans[i] - hubVelVec[i]);

        *tke = f_des_abl * (tkerans) + (1.0 - f_des_abl) * (tkeabl);
        *sdr = f_des_abl * (sdrrans) + (1.0 - f_des_abl) * std::sqrt(tkeabl) /
                                         (0.0856 * std::cbrt(dnv));
      }
    }
  }
}

} // namespace nalu
} // namespace sierra