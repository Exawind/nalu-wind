// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/ContinuityOpenEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
ContinuityOpenEdgeKernel<BcAlgTraits>::ContinuityOpenEdgeKernel(
  const stk::mesh::MetaData& meta,
  SolutionOptions* solnOpts,
  ElemDataRequests& faceData,
  ElemDataRequests& elemData)
  : NGPKernel<ContinuityOpenEdgeKernel<BcAlgTraits>>(),
    solnOpts_(solnOpts),
    coordinates_(get_field_ordinal(meta, solnOpts->get_coordinates_name())),
    velocityRTM_(get_field_ordinal(
      meta, solnOpts->does_mesh_move() ? "velocity_rtm" : "velocity")),
    pressure_(get_field_ordinal(meta, "pressure")),
    density_(get_field_ordinal(meta, "density")),
    exposedAreaVec_(
      get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    pressureBC_(get_field_ordinal(
      meta,
      solnOpts->activateOpenMdotCorrection_ ? "pressure" : "pressure_bc")),
    Gpdx_(get_field_ordinal(meta, "dpdx")),
    Udiag_(get_field_ordinal(meta, "momentum_diag")),
    dynPress_(get_field_ordinal(meta, "dynamic_pressure", meta.side_rank())),
    pstabFac_(solnOpts->activateOpenMdotCorrection_ ? 0.0 : 1.0),
    nocFac_(solnOpts_->get_noc_usage("pressure")),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceData.add_cvfem_face_me(meFC_);
  elemData.add_cvfem_surface_me(meSCS_);

  faceData.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData.add_face_field(dynPress_, BcAlgTraits::numFaceIp_);

  faceData.add_gathered_nodal_field(velocityRTM_, BcAlgTraits::nDim_);
  faceData.add_gathered_nodal_field(density_, 1);
  faceData.add_gathered_nodal_field(pressureBC_, 1);
  faceData.add_gathered_nodal_field(Udiag_, 1);
  faceData.add_gathered_nodal_field(Gpdx_, BcAlgTraits::nDim_);

  elemData.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemData.add_gathered_nodal_field(pressure_, 1);
}

template <typename BcAlgTraits>
void
ContinuityOpenEdgeKernel<BcAlgTraits>::setup(
  const TimeIntegrator& timeIntegrator)
{
  const double dt = timeIntegrator.get_time_step();
  const double gamma1 = timeIntegrator.get_gamma1();
  tauScale_ = dt / gamma1;

  // FIXME: Remove dependence on SolutionOptions
  mdotCorr_ = solnOpts_->activateOpenMdotCorrection_
                ? solnOpts_->mdotAlgOpenCorrection_
                : 0.0;
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
ContinuityOpenEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& faceScratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  const auto& v_velocity = faceScratchViews.get_scratch_view_2D(velocityRTM_);
  const auto& v_density = faceScratchViews.get_scratch_view_1D(density_);
  const auto& v_udiag = faceScratchViews.get_scratch_view_1D(Udiag_);
  const auto& v_pbc = faceScratchViews.get_scratch_view_1D(pressureBC_);
  const auto& v_area = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& v_Gpdx = faceScratchViews.get_scratch_view_2D(Gpdx_);
  const auto& v_dyn_press = faceScratchViews.get_scratch_view_1D(dynPress_);

  const auto& v_pressure = elemScratchViews.get_scratch_view_1D(pressure_);
  const auto& v_coord = elemScratchViews.get_scratch_view_2D(coordinates_);

  for (int ip = 0; ip < BcAlgTraits::nodesPerFace_; ++ip) {
    // Mapping of the nodes to the connected element
    const int nodeR =
      meSCS_->side_node_ordinals(elemFaceOrdinal)[ip]; // nearest node
    const int nodeL =
      meSCS_->opposingNodes(elemFaceOrdinal, ip); // opposing node

    const DoubleType projTimeScale = 1.0 / v_udiag(ip);
    const DoubleType pressureIp = 0.5 * (v_pressure(nodeR) + v_pressure(nodeL));

    DoubleType asq = 0.0;
    DoubleType axdx = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      const DoubleType coordIp = 0.5 * (v_coord(nodeR, d) + v_coord(nodeL, d));
      const DoubleType dxj = v_coord(nodeR, d) - coordIp;
      asq += v_area(ip, d) * v_area(ip, d);
      axdx += v_area(ip, d) * dxj;
    }
    const DoubleType inv_axdx = 1.0 / axdx;

    const DoubleType pbc = v_pbc(ip) - v_dyn_press(ip);
    DoubleType tmdot =
      -projTimeScale * (pbc - pressureIp) * asq * inv_axdx * pstabFac_ -
      mdotCorr_;

    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      const DoubleType coordIp = 0.5 * (v_coord(nodeR, d) + v_coord(nodeL, d));
      const DoubleType axj = v_area(ip, d);
      const DoubleType dxj = v_coord(nodeR, d) - coordIp;
      const DoubleType kxj = axj - asq * inv_axdx * dxj;
      const DoubleType Gjp = v_Gpdx(ip, d);

      tmdot +=
        (v_density(ip) * v_velocity(ip, d) + projTimeScale * Gjp * pstabFac_) *
          axj -
        projTimeScale * kxj * Gjp * nocFac_ * pstabFac_;
    }

    const DoubleType lhsfac =
      0.5 * asq * inv_axdx * pstabFac_ * projTimeScale / tauScale_;

    rhs(nodeR) -= tmdot / tauScale_;
    lhs(nodeR, nodeR) += lhsfac;
    lhs(nodeR, nodeL) += lhsfac;
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ContinuityOpenEdgeKernel)

} // namespace nalu
} // namespace sierra
