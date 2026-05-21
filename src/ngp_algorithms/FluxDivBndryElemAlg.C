// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/FluxDivBndryElemAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/FieldRestriction.hpp"

namespace sierra::kynema_ugf {

template <typename BcAlgTraits>
FluxDivBndryElemAlg<BcAlgTraits>::FluxDivBndryElemAlg(
  Realm& realm,
  stk::mesh::Part* part,
  GenericFieldType* exposed_flux,
  ScalarFieldType* div_flux)
  : Algorithm(realm, part),
    dataNeeded_(realm.meta_data()),
    flux_(exposed_flux->mesh_meta_data_ordinal()),
    div_flux_(div_flux->mesh_meta_data_ordinal()),
    dnv_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(BcAlgTraits::topo_))
{
  dataNeeded_.add_cvfem_face_me(meFC_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(
    coordID, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(dnv_, 1);
  dataNeeded_.add_face_field(flux_, BcAlgTraits::numFaceIp_);
  dataNeeded_.add_master_element_call(FC_SHAPE_FCN, CURRENT_COORDINATES);
}

template <typename BcAlgTraits>
void
FluxDivBndryElemAlg<BcAlgTraits>::execute()
{

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto div_flux = fieldMgr.template get_field<double>(div_flux_);
  const auto div_flux_ops =
    kynema_ugf_ngp::simd_elem_nodal_field_updater(ngpMesh, div_flux);

  const auto dnvID = dnv_;
  const auto fluxID = flux_;
  auto* meFC = meFC_;

  div_flux.sync_to_device();
  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  using view_helper_t = kynema_ugf_ngp::ScalarViewHelper<
    FluxDivBndryElemSimdDataType, ScalarFieldType>;

  const std::string algName =
    (meta.get_fields()[div_flux_]->name() + "_bndry_" +
     std::to_string(BcAlgTraits::topo_));
  kynema_ugf_ngp::run_elem_algorithm(
    algName, meshInfo, meta.side_rank(), dataNeeded_, sel,
    KOKKOS_LAMBDA(typename view_helper_t::SimdDataType & edata) {
      const int* ipNodeMap = meFC->ipNodeMap();
      auto& scrView = edata.simdScrView;
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);
      const auto& v_flux = scrView.get_scratch_view_1D(fluxID);
      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        const int nodeR = ipNodeMap[ip];
        div_flux_ops(edata, ip) += v_flux(ip) / v_dnv(nodeR);
      }
    });
  div_flux.modify_on_device();
}

INSTANTIATE_KERNEL_FACE(FluxDivBndryElemAlg)
} // namespace sierra::kynema_ugf
