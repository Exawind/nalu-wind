/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "user_functions/SteadyThermal3dContactSrcElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <cmath>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
SteadyThermal3dContactSrcElemKernel<AlgTraits>::SteadyThermal3dContactSrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap()),
    a_(1.0),
    k_(1.0),
    pi_(std::acos(-1.0))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);
  get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
SteadyThermal3dContactSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& /* lhs */,
  SharedMemView<DoubleType *>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{

  // Forcing nDim = 3 instead of using AlgTraits::nDim_ here to avoid compiler
  // warnings when this template is instantiated for 2-D topologies. 
  NALU_ALIGNED DoubleType w_scvCoords[3];

  SharedMemView<DoubleType**>& v_coordinates = scratchViews.get_scratch_view_2D(*coordinates_);
  SharedMemView<DoubleType*>& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  // interpolate to ips and evaluate source
  for ( int ip = 0; ip < AlgTraits::numScvIp_; ++ip ) {

    // nearest node to ip
    const int nearestNode = ipNodeMap_[ip];

    // zero out
    for ( int j =0; j < AlgTraits::nDim_; ++j )
      w_scvCoords[j] = 0.0;

    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      for ( int j = 0; j < AlgTraits::nDim_; ++j )
        w_scvCoords[j] += r*v_coordinates(ic,j);
    }

    rhs(nearestNode) += k_/4.0*(2.0*a_*pi_)*(2.0*a_*pi_)*(
      stk::math::cos(2.0*a_*pi_* w_scvCoords[0])
      + stk::math::cos(2.0*a_*pi_* w_scvCoords[1])
      + stk::math::cos(2.0*a_*pi_* w_scvCoords[2]))*v_scv_volume(ip);
  }
}

INSTANTIATE_KERNEL(SteadyThermal3dContactSrcElemKernel)

}  // nalu
}  // sierra
