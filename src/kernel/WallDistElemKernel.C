/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/WallDistElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "TimeIntegrator.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/BulkData.hpp"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
WallDistElemKernel<AlgTraits>::WallDistElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    lrscv_(MasterElementRepo::get_surface_master_element(AlgTraits::topo_)
             ->adjacentNodes()),
    ipNodeMap_(MasterElementRepo::get_volume_master_element(AlgTraits::topo_)
                 ->ipNodeMap()),
    shiftPoisson_(solnOpts.get_shifted_grad_op("ndtw"))
{
  const auto& meta = bulkData.mesh_meta_data();

  coordinates_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());

  MasterElement* meSCS =
    MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  MasterElement* meSCV =
    MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  get_scs_shape_fn_data<AlgTraits>(
    [&](double* ptr) { meSCS->shape_fcn(ptr); }, v_shape_function_);

  dataPreReqs.add_cvfem_surface_me(meSCS);
  dataPreReqs.add_cvfem_volume_me(meSCV);
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);

  auto gradOp = shiftPoisson_ ? SCS_SHIFTED_GRAD_OP : SCS_GRAD_OP;
  dataPreReqs.add_master_element_call(gradOp, CURRENT_COORDINATES);
}

template <typename AlgTraits>
WallDistElemKernel<AlgTraits>::~WallDistElemKernel()
{
}

template <typename AlgTraits>
void
WallDistElemKernel<AlgTraits>::setup(const TimeIntegrator&)
{
}

template <typename AlgTraits>
void
WallDistElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  SharedMemView<DoubleType**>& v_scs_areav =
    scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx =
    shiftPoisson_ ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted
                  : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  // Populate LHS first
  for (int ip = 0; ip < AlgTraits::numScsIp_; ip++) {
    const int il = lrscv_[2 * ip];
    const int ir = lrscv_[2 * ip + 1];

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ic++) {
      DoubleType lhsfac = 0.0;
      for (int j = 0; j < AlgTraits::nDim_; j++)
        lhsfac += v_dndx(ip, ic, j) * v_scs_areav(ip, j);

      lhs(il, ic) -= lhsfac;
      lhs(ir, ic) += lhsfac;
    }
  }

  // Populate RHS next
  for (int ip = 0; ip < AlgTraits::numScvIp_; ip++) {
    const int nearestNode = ipNodeMap_[ip];
    rhs(nearestNode) += v_scv_volume(ip);
  }
}

INSTANTIATE_KERNEL(WallDistElemKernel)

} // namespace nalu
} // namespace sierra
