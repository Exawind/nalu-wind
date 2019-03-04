/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "pmr/RadTransIsotropicScatteringElemKernel.h"
#include "AlgTraits.h"
#include "Enums.h"
#include "master_element/MasterElement.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
RadTransIsotropicScatteringElemKernel<AlgTraits>::RadTransIsotropicScatteringElemKernel(
  const stk::mesh::BulkData& bulkData,
  const bool lumpedMass,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    invFourPi_(1.0/std::acos(-1.0)/4.0),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  scalarFlux_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux");
  scattering_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
 
  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  // compute shape function
  if ( lumpedMass )
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_);
  else
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_gathered_nodal_field(*scalarFlux_, 1);
  dataPreReqs.add_gathered_nodal_field(*scattering_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<typename AlgTraits>
RadTransIsotropicScatteringElemKernel<AlgTraits>::~RadTransIsotropicScatteringElemKernel()
{}

template<typename AlgTraits>
void
RadTransIsotropicScatteringElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>& /* lhs */,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_scalarFlux = scratchViews.get_scratch_view_1D(
    *scalarFlux_);
  SharedMemView<DoubleType*>& v_scattering = scratchViews.get_scratch_view_1D(
    *scattering_);
  SharedMemView<DoubleType*>& v_scvVolume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  // (-scatter*G/4/pi)*dVol = 0.0
  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {

    const int nearestNode = ipNodeMap_[ip];

    DoubleType radSrcScv = 0.0;
    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function_(ip, ic);
      radSrcScv += r*v_scattering(ic)*v_scalarFlux(ic);
    }

    // rhs; lhs n/a
    const DoubleType scvol = v_scvVolume(ip);
    rhs(nearestNode) += radSrcScv*invFourPi_*scvol;
  }
}

INSTANTIATE_KERNEL(RadTransIsotropicScatteringElemKernel)

}  // nalu
}  // sierra
