/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "pmr/RadTransAbsorptionBlackBodyElemKernel.h"
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
RadTransAbsorptionBlackBodyElemKernel<AlgTraits>::RadTransAbsorptionBlackBodyElemKernel(
  const stk::mesh::BulkData& bulkData,
  const bool lumpedMass,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_)->ipNodeMap())
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  intensity_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "intensity");
  absorption_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "absorption_coefficient");
  scattering_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
  radiationSource_ = metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "radiation_source");
  
  MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(AlgTraits::topo_);

  // compute shape function
  if ( lumpedMass )
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shifted_shape_fcn(ptr);}, v_shape_function_);
  else
    get_scv_shape_fn_data<AlgTraits>([&](double* ptr){meSCV->shape_fcn(ptr);}, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // required fields
  dataPreReqs.add_gathered_nodal_field(*intensity_, 1);
  dataPreReqs.add_gathered_nodal_field(*absorption_, 1);
  dataPreReqs.add_gathered_nodal_field(*scattering_, 1);
  dataPreReqs.add_gathered_nodal_field(*radiationSource_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
}

template<typename AlgTraits>
RadTransAbsorptionBlackBodyElemKernel<AlgTraits>::~RadTransAbsorptionBlackBodyElemKernel()
{}

template<typename AlgTraits>
void
RadTransAbsorptionBlackBodyElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType **>&lhs,
  SharedMemView<DoubleType *>&rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_intensity = scratchViews.get_scratch_view_1D(
    *intensity_);
  SharedMemView<DoubleType*>& v_absorption = scratchViews.get_scratch_view_1D(
    *absorption_);
  SharedMemView<DoubleType*>& v_scattering = scratchViews.get_scratch_view_1D(
    *scattering_);
  SharedMemView<DoubleType*>& v_radiationSource = scratchViews.get_scratch_view_1D(
    *radiationSource_);
  SharedMemView<DoubleType*>& v_scvVolume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  // ((mu_a+mu_s)*I - radSrc)*dVol = 0.0
  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {

    const int nearestNode = ipNodeMap_[ip];

    DoubleType iScv = 0.0;
    DoubleType extScv = 0.0;
    DoubleType radSrcScv = 0.0;

    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function_(ip, ic);
      iScv += r*v_intensity(ic);
      extScv += r*(v_absorption(ic) + v_scattering(ic));
      radSrcScv += r*v_radiationSource(ic);
    }

    // rhs
    const DoubleType scvol = v_scvVolume(ip);
    rhs(nearestNode) -= (extScv*iScv - radSrcScv)*scvol;

    // lhs
    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function_(ip, ic);
      lhs(nearestNode,ic) += r*extScv*scvol;
    }
  }
}

INSTANTIATE_KERNEL(RadTransAbsorptionBlackBodyElemKernel)

}  // nalu
}  // sierra
