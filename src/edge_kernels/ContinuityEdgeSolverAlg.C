// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/ContinuityEdgeSolverAlg.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"
#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

ContinuityEdgeSolverAlg::ContinuityEdgeSolverAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  velocity_ = realm.does_mesh_move() && !realm.has_mesh_deformation()
                ? get_field_ordinal(meta, "velocity_rtm")
                : get_field_ordinal(meta, "velocity");
  densityNp1_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  pressure_ = get_field_ordinal(meta, "pressure");
  Gpdx_ = get_field_ordinal(meta, "dpdx");
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  Udiag_ = get_field_ordinal(meta, "momentum_diag");
}

void
ContinuityEdgeSolverAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  // Non-orthogonal correction factor for continuity equation system
  const std::string dofName = "pressure";
  const DblType nocFac = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // Classic Nalu projection timescale
  const DblType dt = realm_.get_time_step();
  const DblType gamma1 = realm_.get_gamma1();
  const DblType tauScale = dt / gamma1;

  // Interpolation option for rho*U
  const DblType interpTogether = realm_.get_mdot_interp();
  const DblType om_interpTogether = (1.0 - interpTogether);

  const DblType solveIncompressibleEqn = realm_.get_incompressible_solve();
  const DblType om_solveIncompressibleEqn = 1.0 - solveIncompressibleEqn;

  const bool add_balanced_forcing =
    realm_.solutionOptions_->use_balanced_buoyancy_force_;

  DblType gravity[3] = {};
  if (add_balanced_forcing) {
    const auto& solnOptsGravity =
      realm_.solutionOptions_->get_gravity_vector(ndim);
    for (int idim = 0; idim < ndim; ++idim) {
      gravity[idim] = solnOptsGravity[idim];
    }
  }

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto coordinates = fieldMgr.get_field<double>(coordinates_);
  auto velocity = fieldMgr.get_field<double>(velocity_);
  auto Gpdx = fieldMgr.get_field<double>(Gpdx_);
  auto density = fieldMgr.get_field<double>(densityNp1_);
  auto pressure = fieldMgr.get_field<double>(pressure_);
  auto udiag = fieldMgr.get_field<double>(Udiag_);
  auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);

  auto source = !add_balanced_forcing
                  ? fieldMgr.get_field<double>(Gpdx_)
                  : fieldMgr.get_field<double>(
                      get_field_ordinal(realm_.meta_data(), "buoyancy_source"));

  auto source_mask = !add_balanced_forcing
                       ? fieldMgr.get_field<double>(densityNp1_)
                       : fieldMgr.get_field<double>(get_field_ordinal(
                           realm_.meta_data(), "buoyancy_source_mask"));

  stk::mesh::NgpField<double> edgeFaceVelMag;
  bool needs_gcl = false;
  if (realm_.has_mesh_deformation()) {
    needs_gcl = true;
    edgeFaceVelMag_ = get_field_ordinal(
      realm_.meta_data(), "edge_face_velocity_mag", stk::topology::EDGE_RANK);
    edgeFaceVelMag = fieldMgr.get_field<double>(edgeFaceVelMag_);
    edgeFaceVelMag.sync_to_device();
  }
  coordinates.sync_to_device();
  velocity.sync_to_device();
  Gpdx.sync_to_device();
  density.sync_to_device();
  pressure.sync_to_device();
  udiag.sync_to_device();
  edgeAreaVec.sync_to_device();
  source.sync_to_device();
  source_mask.sync_to_device();

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
       DblType av[NDimMax_];

      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      const DblType pressureL = pressure.get(nodeL, 0);
      const DblType pressureR = pressure.get(nodeR, 0);

      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);

      const DblType udiagL = udiag.get(nodeL, 0);
      const DblType udiagR = udiag.get(nodeR, 0);
      const DblType projTimeScale = 0.5 * (1.0 / udiagL + 1.0 / udiagR);
      const DblType rhoIp = 0.5 * (densityL + densityR);
      const DblType denScale =
        (1.0 / rhoIp) * solveIncompressibleEqn + om_solveIncompressibleEqn;

      DblType axdx = 0.0;
      DblType asq = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      DblType tmdot = -projTimeScale * (pressureR - pressureL) * asq * inv_axdx;

      if (add_balanced_forcing) {
        const DblType masked_weights =
          0.5 * (source_mask.get(nodeL, 0) + source_mask.get(nodeR, 0));
        for (int d = 0; d < ndim; ++d) {
          tmdot += projTimeScale * av[d] * gravity[d] * rhoIp * masked_weights;
        }
      }
      if (needs_gcl) {
        tmdot -= rhoIp * edgeFaceVelMag.get(edge, 0);
      }

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        // non-orthogonal correction
        const DblType kxj = av[d] - asq * inv_axdx * dxj;
        const DblType rhoUjIp = 0.5 * (densityR * velocity.get(nodeR, d) +
                                       densityL * velocity.get(nodeL, d));
        const DblType ujIp =
          0.5 * (velocity.get(nodeR, d) + velocity.get(nodeL, d));
        DblType GjIp =
          0.5 * (Gpdx.get(nodeR, d) / (udiagR) + Gpdx.get(nodeL, d) / (udiagL));
        if (add_balanced_forcing) {
          GjIp -=
            0.5 *
            ((source_mask.get(nodeR, 0) * source.get(nodeR, d)) / (udiagR) +
             (source_mask.get(nodeL, 0) * source.get(nodeL, d)) / (udiagL));
        }
        tmdot +=
          (interpTogether * rhoUjIp + om_interpTogether * rhoIp * ujIp + GjIp) *
            av[d] -
          kxj * GjIp * nocFac;
      }
      tmdot /= tauScale;
      tmdot *= denScale;
      const DblType lhsfac =
        -asq * inv_axdx * projTimeScale * denScale / tauScale;

      // Left node entries
      smdata.lhs(0, 0) = -lhsfac;
      smdata.lhs(0, 1) = +lhsfac;
      smdata.rhs(0) = -tmdot;

      // Right node entries
      smdata.lhs(1, 0) = +lhsfac;
      smdata.lhs(1, 1) = -lhsfac;
      smdata.rhs(1) = tmdot;
    });
}

} // namespace nalu
} // namespace sierra
