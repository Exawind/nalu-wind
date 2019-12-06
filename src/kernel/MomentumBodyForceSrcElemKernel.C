// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

 
#include "kernel/MomentumBodyForceSrcElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MomentumBodyForceSrcElemKernel<AlgTraits>::MomentumBodyForceSrcElemKernel(
  const stk::mesh::BulkData& /* bulkData */,
  const SolutionOptions& /*solnOpts*/,
  const std::vector<double>& params,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(
                 AlgTraits::topo_)
                 ->ipNodeMap())
{

  for (int i = 0; i < AlgTraits::nDim_; i++)
    bodyForce_(i) = params[i];

  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(
      AlgTraits::topo_);
  get_scv_shape_fn_data<AlgTraits>(
    [&](double* ptr) { meSCV->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template <typename AlgTraits>
MomentumBodyForceSrcElemKernel<AlgTraits>::~MomentumBodyForceSrcElemKernel()
{
}

template <typename AlgTraits>
void
MomentumBodyForceSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /* lhs */,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap_[ip];

    // Compute RHS
    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      rhs(nnNdim + j) += scV * bodyForce_(j);
    }

    // No LHS contributions
  }
}

INSTANTIATE_KERNEL(MomentumBodyForceSrcElemKernel)

} // namespace nalu
} // namespace sierra
