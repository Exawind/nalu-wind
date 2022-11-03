// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/ScalarEdgeOpenSolverAlg.h"
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
ScalarEdgeOpenSolverAlg<BcAlgTraits>::ScalarEdgeOpenSolverAlg(
  const stk::mesh::MetaData& meta,
  const SolutionOptions& solnOpts,
  ScalarFieldType* scalarQ,
  ScalarFieldType* bcScalarQ,
  VectorFieldType* dqdx,
  ScalarFieldType* diffFluxCoeff,
  ElemDataRequests& faceDataPreReqs,
  ElemDataRequests& elemDataPreReqs)
  : NGPKernel<ScalarEdgeOpenSolverAlg<BcAlgTraits>>(),
    scalarQ_(
      scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal()),
    bcScalarQ_(bcScalarQ->mesh_meta_data_ordinal()),
    dqdx_(dqdx->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    coordinates_(get_field_ordinal(meta, solnOpts.get_coordinates_name())),
    openMassFlowRate_(
      get_field_ordinal(meta, "open_mass_flow_rate", meta.side_rank())),
    relaxFac_(solnOpts.get_relaxation_factor(scalarQ->name())),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(sierra::nalu::MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceDataPreReqs.add_cvfem_face_me(meFC_);
  elemDataPreReqs.add_cvfem_surface_me(meSCS_);

  faceDataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  faceDataPreReqs.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);
  elemDataPreReqs.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemDataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  elemDataPreReqs.add_gathered_nodal_field(bcScalarQ_, 1);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
ScalarEdgeOpenSolverAlg<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& faceScratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  // Mapping of face nodes into element indices
  const int* ipNodeMap = meSCS_->ipNodeMap(elemFaceOrdinal);

  // Field variables on the boundary face
  auto& v_mdot = faceScratchViews.get_scratch_view_1D(openMassFlowRate_);

  // Field variables on element connected to the boundary face
  auto& v_scalarQ = elemScratchViews.get_scratch_view_1D(scalarQ_);
  auto& v_bcScalarQ = elemScratchViews.get_scratch_view_1D(bcScalarQ_);

  for (int ip = 0; ip < BcAlgTraits::nodesPerFace_; ++ip) {
    // ip is the index of the node in the face array
    const int nodeR = ipNodeMap[ip];

    const DoubleType qR = v_scalarQ(nodeR);
    const DoubleType qEntrain = v_bcScalarQ(nodeR);

    const DoubleType mdot = v_mdot(ip);

    //================================
    // advection first (and only)
    //================================

    // Account for both total advection leaving the domain and entrainment
    const DoubleType aflux =
      stk::math::if_then_else(mdot > 0.0, mdot * qR, mdot * qEntrain);
    const DoubleType upwind =
      stk::math::if_then_else(mdot > 0.0, mdot / relaxFac_, 0.0);

    rhs(nodeR) -= aflux;

    lhs(nodeR, nodeR) += upwind;
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ScalarEdgeOpenSolverAlg)

} // namespace nalu
} // namespace sierra
