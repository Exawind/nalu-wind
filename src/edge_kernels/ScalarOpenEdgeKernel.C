// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/ScalarOpenEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
ScalarOpenEdgeKernel<BcAlgTraits>::ScalarOpenEdgeKernel(
  const stk::mesh::MetaData& meta,
  const SolutionOptions& solnOpts,
  ScalarFieldType* scalarQ,
  ScalarFieldType* bcScalarQ,
  ElemDataRequests& faceData)
  : NGPKernel<ScalarOpenEdgeKernel<BcAlgTraits>>(),
    scalarQ_(
      scalarQ->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal()),
    bcScalarQ_(bcScalarQ->mesh_meta_data_ordinal()),
    openMassFlowRate_(
      get_field_ordinal(meta, "open_mass_flow_rate", meta.side_rank())),
    relaxFac_(solnOpts.get_relaxation_factor(scalarQ->name())),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          BcAlgTraits>())
{
  faceData.add_cvfem_face_me(meFC_);

  faceData.add_gathered_nodal_field(scalarQ_, 1);
  faceData.add_gathered_nodal_field(bcScalarQ_, 1);
  faceData.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
ScalarOpenEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const int* ipNodeMap = meFC_->ipNodeMap();
  const auto& v_mdot = scratchViews.get_scratch_view_1D(openMassFlowRate_);
  const auto& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  const auto& v_bcScalarQ = scratchViews.get_scratch_view_1D(bcScalarQ_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nodeR = ipNodeMap[ip];

    const DoubleType qR = v_scalarQ(nodeR);
    const DoubleType qEntrain = v_bcScalarQ(nodeR);
    const DoubleType mdot = v_mdot(ip);

    const DoubleType uUpw = stk::math::if_then_else((mdot > 0.0), qR, qEntrain);
    const DoubleType lhsfac = stk::math::if_then_else((mdot > 0.0), 1.0, 0.0);

    rhs(nodeR) -= mdot * uUpw;
    lhs(nodeR, nodeR) += lhsfac * mdot / relaxFac_;
  }
}

INSTANTIATE_KERNEL_FACE(ScalarOpenEdgeKernel)

} // namespace nalu
} // namespace sierra
