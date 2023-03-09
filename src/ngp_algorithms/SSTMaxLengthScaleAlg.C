// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/SSTMaxLengthScaleAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_algorithms/ViewHelper.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
SSTMaxLengthScaleAlg<AlgTraits>::SSTMaxLengthScaleAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    maxLengthScale_(
      get_field_ordinal(realm_.meta_data(), "sst_max_length_scale")),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{
}

template <typename AlgTraits>
void
SSTMaxLengthScaleAlg<AlgTraits>::execute()
{
  using ElemInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto coordinates = fieldMgr.template get_field<double>(coordinates_);
  auto maxLengthScale = fieldMgr.template get_field<double>(maxLengthScale_);
  MasterElement* meSCS = meSCS_;

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const std::string algName =
    "compute_sst_max_length_scale_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, ngpMesh, stk::topology::ELEM_RANK, sel,
    KOKKOS_LAMBDA(const ElemInfoType& einfo) {
      const stk::mesh::Entity entity = einfo.entity;
      const auto nodes = ngpMesh.get_nodes(
        stk::topology::ELEM_RANK, ngpMesh.fast_mesh_index(entity));

      const int* lrscv = meSCS->adjacentNodes();
      for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
        // left and right nodes for this ip
        const int il = lrscv[2 * ip];
        const int ir = lrscv[2 * ip + 1];

        const auto nodeL = ngpMesh.fast_mesh_index(nodes[il]);
        const auto nodeR = ngpMesh.fast_mesh_index(nodes[ir]);

        double dx = 0;
        for (int d = 0; d < AlgTraits::nDim_; ++d) {
          const double dxj =
            coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
          dx += dxj * dxj;
        }
        dx = stk::math::sqrt(dx);
        double& maxLengthL = maxLengthScale.get(nodeL, 0);
        double& maxLengthR = maxLengthScale.get(nodeR, 0);

        if (maxLengthL < dx)
          Kokkos::atomic_add(&maxLengthL, dx - maxLengthL);
        if (maxLengthR < dx)
          Kokkos::atomic_add(&maxLengthR, dx - maxLengthR);
      }
    });
  maxLengthScale.modify_on_device();
}

INSTANTIATE_KERNEL(SSTMaxLengthScaleAlg)

} // namespace nalu
} // namespace sierra
