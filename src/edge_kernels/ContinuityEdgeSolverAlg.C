/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "edge_kernels/ContinuityEdgeSolverAlg.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

ContinuityEdgeSolverAlg::ContinuityEdgeSolverAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem
) : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();
  const int ndim = meta.spatial_dimension();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string velField = realm.does_mesh_move()? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(meta, velField);
  densityNp1_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  pressure_ = get_field_ordinal(meta, "pressure");
  Gpdx_ = get_field_ordinal(meta, "dpdx");
  edgeAreaVec_ = get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  Udiag_ = get_field_ordinal(meta, "momentum_diag");

  dataNeeded_.add_coordinates_field(coordinates_, ndim, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(velocityRTM_, ndim);
  dataNeeded_.add_gathered_nodal_field(densityNp1_, 1);
  dataNeeded_.add_gathered_nodal_field(pressure_, 1);
  dataNeeded_.add_gathered_nodal_field(Udiag_, 1);
  dataNeeded_.add_gathered_nodal_field(Gpdx_, ndim);
}

void
ContinuityEdgeSolverAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  // Non-orthogonal correction factor for continuity equation system
  const std::string dofName = "pressure";
  const DblType nocFac
    = (realm_.get_noc_usage(dofName) == true) ? 1.0 : 0.0;

  // Classic Nalu projection timescale
  const DblType dt = realm_.get_time_step();
  const DblType gamma1 = realm_.get_gamma1();
  const DblType tauScale = dt / gamma1;

  // Interpolation option for rho*U
  const DblType interpTogether = realm_.get_mdot_interp();
  const DblType om_interpTogether = (1.0 - interpTogether);

  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(ShmemDataType& smdata, const stk::mesh::FastMeshIndex & edge)
    {
      auto& scrViews = smdata.preReqData;
      const auto& v_coords = scrViews.get_scratch_view_2D(coordinates_);
      const auto& v_velocity = scrViews.get_scratch_view_2D(velocityRTM_);
      const auto& v_Gpdx = scrViews.get_scratch_view_2D(Gpdx_);
      const auto& v_density = scrViews.get_scratch_view_1D(densityNp1_);
      const auto& v_pressure = scrViews.get_scratch_view_1D(pressure_);
      const auto& v_udiag = scrViews.get_scratch_view_1D(Udiag_);

      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[nDimMax_];
      for (int d=0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      const DblType projTimeScale = 0.5 * (1.0 / v_udiag(nodeL) + 1.0 / v_udiag(nodeR));
      const DblType rhoIp = 0.5 * (v_density(nodeL) + v_density(nodeR));

      // Compute geometry
      DblType axdx = 0.0;
      DblType asq = 0.0;
      for (int d=0; d < ndim; ++d) {
        const DblType dxj = v_coords(nodeR, d) - v_coords(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      DblType tmdot = -projTimeScale * (v_pressure(nodeR) - v_pressure(nodeL)) *
                      asq * inv_axdx;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj = v_coords(nodeR, d) - v_coords(nodeL, d);
        // non-orthogonal correction
        const DblType kxj = av[d] - asq * inv_axdx * dxj;
        const DblType rhoUjIp = 0.5 * (v_density(nodeR) * v_velocity(nodeR, d) +
                                       v_density(nodeL) * v_velocity(nodeL, d));
        const DblType ujIp = 0.5 * (v_velocity(nodeL, d) + v_velocity(nodeR, d));
        const DblType GjIp = 0.5 * (v_Gpdx(nodeR, d) / v_udiag(nodeR) +
                                    v_Gpdx(nodeL, d) / v_udiag(nodeL));

        tmdot += (interpTogether * rhoUjIp +
                  om_interpTogether * rhoIp * ujIp + GjIp) * av[d]
          - kxj * GjIp * nocFac;
      }

      tmdot /= tauScale;
      const DblType lhsfac = -asq * inv_axdx * projTimeScale / tauScale;

      // Left node entries
      smdata.lhs(nodeL, nodeL) = -lhsfac;
      smdata.lhs(nodeL, nodeR) = +lhsfac;
      smdata.rhs(nodeL) = -tmdot;

      // Right node entries
      smdata.lhs(nodeR, nodeL) = +lhsfac;
      smdata.lhs(nodeR, nodeR) = -lhsfac;
      smdata.rhs(nodeR) = tmdot;

#ifndef KOKKOS_ENABLE_CUDA
      apply_coeff(
        nodesPerEntity_, smdata.ngpElemNodes, smdata.scratchIds,
        smdata.sortPermutation, smdata.rhs, smdata.lhs, __FILE__);
#endif
    });
}

}  // nalu
}  // sierra
