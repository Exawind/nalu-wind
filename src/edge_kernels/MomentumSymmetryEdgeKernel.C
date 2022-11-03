// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "SimdInterface.h"
#include "edge_kernels/MomentumSymmetryEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
MomentumSymmetryEdgeKernel<BcAlgTraits>::MomentumSymmetryEdgeKernel(
  const stk::mesh::MetaData& meta,
  const SolutionOptions& solnOpts,
  VectorFieldType* velocity,
  ScalarFieldType* viscosity,
  ElemDataRequests& faceDataPreReqs,
  ElemDataRequests& elemDataPreReqs)
  : NGPKernel<MomentumSymmetryEdgeKernel<BcAlgTraits>>(),
    coordinates_(get_field_ordinal(meta, solnOpts.get_coordinates_name())),
    velocityNp1_(
      velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal()),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    exposedAreaVec_(
      get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    dudx_(get_field_ordinal(meta, "dudx")),
    includeDivU_(solnOpts.includeDivU_),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>()),
    penaltyFactor_(solnOpts.symmetryBcPenaltyFactor_)
{
  faceDataPreReqs.add_cvfem_face_me(meFC_);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  faceDataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  faceDataPreReqs.add_gathered_nodal_field(
    dudx_, BcAlgTraits::nDim_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  elemDataPreReqs.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
MomentumSymmetryEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& faceScratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  // Work arrays
  NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType duidxj[BcAlgTraits::nDim_][BcAlgTraits::nDim_];

  // Field variables on the boundary face
  auto& v_visc = faceScratchViews.get_scratch_view_1D(viscosity_);
  auto& v_dudx = faceScratchViews.get_scratch_view_3D(dudx_);
  auto& v_areavec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);

  // Field variables on element connected to the boundary face
  auto& v_uNp1 = elemScratchViews.get_scratch_view_2D(velocityNp1_);
  auto& v_coords = elemScratchViews.get_scratch_view_2D(coordinates_);

  // Mapping of face nodes into element indices
  const int* ipNodeMap = meSCS_->ipNodeMap(elemFaceOrdinal);

  for (int ip = 0; ip < BcAlgTraits::nodesPerFace_; ++ip) {
    // ip is the index of the node in the face array, nodeL and nodeR are the
    // indices for the face node and opposing node in the element arrays
    const int nodeR = ipNodeMap[ip];
    const int nodeL = meSCS_->opposingNodes(elemFaceOrdinal, ip);

    // Extract viscosity for this node from face data
    const auto visc = v_visc(ip);

    // Compute area vector related quantities
    DoubleType axdx = 0.0;
    DoubleType asq = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      const DoubleType dxj = v_coords(nodeR, d) - v_coords(nodeL, d);
      asq += v_areavec(ip, d) * v_areavec(ip, d);
      axdx += v_areavec(ip, d) * dxj;
    }
    const DoubleType inv_axdx = 1.0 / axdx;

    // Populate unit vector for later use
    const DoubleType amag = stk::math::sqrt(asq);
    for (int d = 0; d < BcAlgTraits::nDim_; ++d)
      nx[d] = v_areavec(ip, d) / amag;

    // Computation of duidxj term, reproduce original comment by S. P. Domino
    /*
      form duidxj with over-relaxed procedure of Jasak:

      dui/dxj = GjUi +[(uiR - uiL) - GlUi*dxl]*Aj/AxDx
      where Gp is the interpolated pth nodal gradient for ui
    */
    for (int i = 0; i < BcAlgTraits::nDim_; i++) {
      const auto dui = v_uNp1(nodeR, i) - v_uNp1(nodeL, i);

      // Non-orthogonal correction
      DoubleType gjuidx = 0.0;
      for (int j = 0; j < BcAlgTraits::nDim_; j++) {
        const DoubleType dxj = v_coords(nodeR, j) - v_coords(nodeL, j);
        gjuidx += v_dudx(ip, i, j) * dxj;
      }

      // final dui/dxj with non-orthogonal contributions
      for (int j = 0; j < BcAlgTraits::nDim_; j++) {
        duidxj[i][j] =
          v_dudx(ip, i, j) + (dui - gjuidx) * v_areavec(ip, j) * inv_axdx;
      }
    }

    // div(U) terms
    DoubleType divU = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d)
      divU += duidxj[d][d];

    // Viscous flux terms
    DoubleType fxnx = 0.0;
    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      DoubleType fxi =
        2.0 / 3.0 * visc * divU * v_areavec(ip, i) * includeDivU_;

      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        fxi += -visc * (duidxj[i][j] + duidxj[j][i]) * v_areavec(ip, j);
      }

      fxnx += nx[i] * fxi;
    }

    const DoubleType diffusionMassRate = -visc * asq * inv_axdx;
    const DoubleType penaltyFac = penaltyFactor_ * diffusionMassRate;
    DoubleType uN = 0;
    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      uN += v_uNp1(nodeR, i) * nx[i];
    }

    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      const int rowR = nodeR * BcAlgTraits::nDim_ + i;
      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        const int colR = nodeR * BcAlgTraits::nDim_ + j;
        lhs(rowR, colR) -= penaltyFac * nx[i] * nx[j];
      }
      rhs(rowR) += penaltyFac * uN * nx[i];
    }

    for (int i = 0; i < BcAlgTraits::nDim_; i++) {
      const int rowL = nodeL * BcAlgTraits::nDim_ + i;
      const int rowR = nodeR * BcAlgTraits::nDim_ + i;

      rhs(rowR) -= fxnx * nx[i];
      DoubleType lhsfac = diffusionMassRate * nx[i] * nx[i];
      lhs(rowR, rowL) -= lhsfac;
      lhs(rowR, rowR) += lhsfac;

      const DoubleType axi = v_areavec(ip, i);
      const DoubleType nxnx = nx[i] * nx[i];

      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        const int colL = nodeL * BcAlgTraits::nDim_ + j;
        const int colR = nodeR * BcAlgTraits::nDim_ + j;

        const DoubleType axj = v_areavec(ip, j);
        lhsfac = -visc * axi * axj * inv_axdx * nxnx;
        lhs(rowR, colL) -= lhsfac;
        lhs(rowR, colR) += lhsfac;

        if (i == j)
          continue;

        lhsfac = -visc * asq * inv_axdx * nx[i] * nx[j];
        lhs(rowR, colL) -= lhsfac;
        lhs(rowR, colR) += lhsfac;

        lhsfac = -visc * axj * axj * nx[i] * nx[j];
        lhs(rowR, colL) -= lhsfac;
        lhs(rowR, colR) += lhsfac;

        lhsfac = -visc * axj * axi * nx[i] * nx[j];
        lhs(rowR, rowL) -= lhsfac;
        lhs(rowR, rowR) += lhsfac;
      }
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumSymmetryEdgeKernel)

} // namespace nalu
} // namespace sierra
