// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "Enums.h"
#include "edge_kernels/MomentumOpenEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include <stk_math/StkMath.hpp>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- Constructor for MomentumOpenEdgeKernel --------------------------
//--------------------------------------------------------------------------
template <typename BcAlgTraits>
MomentumOpenEdgeKernel<BcAlgTraits>::MomentumOpenEdgeKernel(
  const stk::mesh::MetaData& meta,
  SolutionOptions* solnOpts,
  ScalarFieldType* viscosity,
  ElemDataRequests& faceData,
  ElemDataRequests& elemData,
  EntrainmentMethod method)
  : NGPKernel<MomentumOpenEdgeKernel<BcAlgTraits>>(),
    coordinates_(get_field_ordinal(meta, solnOpts->get_coordinates_name())),
    dudx_(get_field_ordinal(meta, "dudx")),
    exposedAreaVec_(
      get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    openMassFlowRate_(
      get_field_ordinal(meta, "open_mass_flow_rate", meta.side_rank())),
    velocityBc_(get_field_ordinal(meta, "open_velocity_bc")),
    velocityNp1_(get_field_ordinal(meta, "velocity", stk::mesh::StateNP1)),
    viscosity_(viscosity->mesh_meta_data_ordinal()),
    includeDivU_(solnOpts->includeDivU_),
    nfEntrain_(solnOpts->nearestFaceEntrain_),
    entrain_(method),
    turbModel_(solnOpts->turbulenceModel_),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceData.add_cvfem_face_me(meFC_);
  elemData.add_cvfem_surface_me(meSCS_);

  faceData.add_gathered_nodal_field(
    dudx_, BcAlgTraits::nDim_, BcAlgTraits::nDim_);
  faceData.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);
  faceData.add_gathered_nodal_field(velocityBc_, BcAlgTraits::nDim_);
  faceData.add_gathered_nodal_field(viscosity_, 1);

  elemData.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemData.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
template <typename BcAlgTraits>
KOKKOS_FUNCTION void
MomentumOpenEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& faceScratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  // nearest face entrainment
  const double om_nfEntrain = 1.0 - nfEntrain_;

  // Work arrays
  NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType fx[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType duidxj[BcAlgTraits::nDim_][BcAlgTraits::nDim_];

  // Field variables on the boundary face
  auto& v_areavec = faceScratchViews.get_scratch_view_2D(exposedAreaVec_);
  auto& v_dudx = faceScratchViews.get_scratch_view_3D(dudx_);
  auto& v_massflow = faceScratchViews.get_scratch_view_1D(openMassFlowRate_);
  auto& v_uBc = faceScratchViews.get_scratch_view_2D(velocityBc_);
  auto& v_visc = faceScratchViews.get_scratch_view_1D(viscosity_);

  // Field variables on element connected to the boundary face
  auto& v_coords = elemScratchViews.get_scratch_view_2D(coordinates_);
  auto& v_uNp1 = elemScratchViews.get_scratch_view_2D(velocityNp1_);

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
    const DoubleType amag = stk::math::sqrt(asq);

    // Populate unit normal vector and other velocity projections
    // for later use
    DoubleType uxnx = 0.0;
    DoubleType uxnxip = 0.0;
    DoubleType uspecxnx = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      nx[d] = v_areavec(ip, d) / amag;
      uxnx += nx[d] * v_uNp1(nodeR, d);
      uxnxip += 0.5 * nx[d] * (v_uNp1(nodeL, d) + v_uNp1(nodeR, d));
      uspecxnx += nx[d] * v_uBc(ip, d);
    }

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

    // div(U)
    DoubleType divU = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d)
      divU += duidxj[d][d];

    // Viscous forces
    DoubleType fxnx = 0.0;
    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      fx[i] = 2.0 / 3.0 * visc * divU * v_areavec(ip, i) * includeDivU_;

      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        fx[i] += -visc * (duidxj[i][j] + duidxj[j][i]) * v_areavec(ip, j);
      }
      fxnx += nx[i] * fx[i];
    }

    // full stress, sigma_ij
    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      const int rowL = nodeL * BcAlgTraits::nDim_ + i;
      const int rowR = nodeR * BcAlgTraits::nDim_ + i;

      const DoubleType axi = v_areavec(ip, i);

      // subtract normal component of the flux
      rhs(rowR) -= (fx[i] - nx[i] * fxnx);

      const DoubleType om_nxinxi = 1.0 - nx[i] * nx[i];
      DoubleType lhsFac = -visc * asq * inv_axdx * om_nxinxi;
      lhs(rowR, rowL) -= lhsFac;
      lhs(rowR, rowR) += lhsFac;

      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        const int colL = nodeL * BcAlgTraits::nDim_ + j;
        const int colR = nodeR * BcAlgTraits::nDim_ + j;

        const DoubleType axj = v_areavec(ip, j);
        lhsFac = -visc * axi * axj * inv_axdx * om_nxinxi;
        lhs(rowR, colL) -= lhsFac;
        lhs(rowR, colR) += lhsFac;

        if (i == j)
          continue;

        const DoubleType nxinxj = nx[i] * nx[j];

        lhsFac = visc * asq * inv_axdx * nxinxj;
        lhs(rowR, colL) -= lhsFac;
        lhs(rowR, colR) += lhsFac;

        lhsFac = visc * axj * axj * inv_axdx * nxinxj;
        lhs(rowR, colL) -= lhsFac;
        lhs(rowR, colR) += lhsFac;

        lhsFac = visc * axj * axi * inv_axdx * nxinxj;
        lhs(rowR, rowL) -= lhsFac;
        lhs(rowR, rowR) += lhsFac;
      }
    }

    switch (entrain_) {
    case EntrainmentMethod::SPECIFIED: {
      const auto tmdot = v_massflow(ip);
      for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
        const int rowR = nodeR * BcAlgTraits::nDim_ + i;
        const auto sigma = visc * asq * inv_axdx;
        const auto lambda =
          0.5 * (tmdot - stk::math::sqrt(tmdot * tmdot + 8 * sigma * sigma));
        rhs(rowR) -= stk::math::if_then_else(
          tmdot > 0, tmdot * v_uNp1(nodeR, i),
          tmdot * v_uNp1(nodeR, i) -
            lambda * (v_uNp1(nodeR, i) - v_uBc(ip, i)));
        lhs(rowR, rowR) +=
          stk::math::if_then_else(tmdot > 0, tmdot, tmdot - lambda);
      }
      break;
    }
    case EntrainmentMethod::COMPUTED: {
      // advection
      const DoubleType tmdot = v_massflow(ip);

      for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
        const int rowR = nodeR * BcAlgTraits::nDim_ + i;

        rhs(rowR) -= stk::math::if_then_else(
          (tmdot > 0.0), tmdot * v_uNp1(nodeR, i), // leaving the domain
          tmdot * ((nfEntrain_ * uxnx + om_nfEntrain * uxnxip) *
                     nx[i] + // constrain to be normal
                   (v_uBc(ip, i) -
                    uspecxnx * nx[i]))); // user spec entrainment (tangential)

        // leaving the domain
        lhs(rowR, rowR) += stk::math::if_then_else((tmdot > 0.0), tmdot, 0.0);

        // entraining; constrain to be normal
        for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
          const int colL = nodeL * BcAlgTraits::nDim_ + j;
          const int colR = nodeR * BcAlgTraits::nDim_ + j;

          lhs(rowR, colL) += stk::math::if_then_else(
            (tmdot > 0.0), 0.0, tmdot * om_nfEntrain * 0.5 * nx[i] * nx[j]);
          lhs(rowR, colR) += stk::math::if_then_else(
            (tmdot > 0.0), 0.0,
            tmdot * (nfEntrain_ + om_nfEntrain * 0.5) * nx[i] * nx[j]);
        }
      }
      break;
    }
    default:
      NGP_ThrowErrorMsg("invalid entrainment method");
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumOpenEdgeKernel)

} // namespace nalu
} // namespace sierra
