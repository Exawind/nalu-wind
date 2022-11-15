// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/VOFAdvectionEdgeAlg.h"
#include "EquationSystem.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "edge_kernels/EdgeKernelUtils.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

VOFAdvectionEdgeAlg::VOFAdvectionEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  ScalarFieldType* scalarQ,
  VectorFieldType* dqdx,
  const bool useAverages)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string vrtmName =
    realm.does_mesh_move() ? "velocity_rtm" : "velocity";
  const std::string avgVrtmName =
    realm.does_mesh_move() ? "average_velocity_rtm" : "average_velocity";

  scalarQ_ = scalarQ->mesh_meta_data_ordinal();
  dqdx_ = dqdx->mesh_meta_data_ordinal();
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  massFlowRate_ = get_field_ordinal(
    meta, (useAverages) ? "average_mass_flow_rate" : "mass_flow_rate",
    stk::topology::EDGE_RANK);
  velocityRTM_ =
    get_field_ordinal(meta, (useAverages) ? avgVrtmName : vrtmName);
}

void
VOFAdvectionEdgeAlg::execute()
{
  const double eps = 1.0e-16;
  const int ndim = realm_.meta_data().spatial_dimension();

  const DblType alpha = realm_.get_alpha_factor("volume_of_fluid");
  const DblType alphaUpw = realm_.get_alpha_upw_factor("volume_of_fluid");
  const DblType hoUpwind = realm_.get_upw_factor("volume_of_fluid");
  const DblType relaxFac =
    realm_.solutionOptions_->get_relaxation_factor("volume_of_fluid");
  const bool useLimiter = realm_.primitive_uses_limiter("volume_of_fluid");

  const DblType om_alpha = 1.0 - alpha;
  const DblType om_alphaUpw = 1.0 - alphaUpw;

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vrtm = fieldMgr.get_field<double>(velocityRTM_);
  const auto scalarQ = fieldMgr.get_field<double>(scalarQ_);
  const auto dqdx = fieldMgr.get_field<double>(dqdx_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[NDimMax_];
      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      const DblType mdot = massFlowRate.get(edge, 0);

      const DblType qNp1L = scalarQ.get(nodeL, 0);
      const DblType qNp1R = scalarQ.get(nodeR, 0);

      // Compute area vector related quantities and (U dot areaVec)
      DblType axdx = 0.0;
      DblType asq = 0.0;
      DblType udotx = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
        udotx += 0.5 * dxj * (vrtm.get(nodeR, d) + vrtm.get(nodeL, d));
      }
      const DblType inv_axdx = 1.0 / axdx;

      // Advective flux
      const DblType qIp = 0.5 * (qNp1R + qNp1L); // 2nd order central term

      // Upwinded term
      const DblType qUpw = (mdot > 0) ? (alphaUpw * qNp1L + om_alphaUpw * qIp)
                                      : (alphaUpw * qNp1R + om_alphaUpw * qIp);

      const DblType adv_flux = mdot * qUpw;
      smdata.rhs(0) -= adv_flux;
      smdata.rhs(1) += adv_flux;

      // Left node contribution; upwind terms
      DblType alhsfac = 0.5 * (mdot + stk::math::abs(mdot)) * alphaUpw;
      smdata.lhs(0, 0) += alhsfac / relaxFac;
      smdata.lhs(1, 0) -= alhsfac;

      // Right node contribution; upwind terms
      alhsfac = 0.5 * (mdot - stk::math::abs(mdot)) * alphaUpw;
      smdata.lhs(1, 1) -= alhsfac / relaxFac;
      smdata.lhs(0, 1) += alhsfac;
    });
}

} // namespace nalu
} // namespace sierra
