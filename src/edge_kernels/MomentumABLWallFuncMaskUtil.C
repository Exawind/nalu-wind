// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/MomentumABLWallFuncMaskUtil.h>
#include <Realm.h>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpLoopUtils.h>
#include <ngp_utils/NgpTypes.h>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>

// Extra crap nmatula
#include <edge_kernels/MomentumEdgePecletAlg.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <Enums.h>
#include <string>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpTypes.h>
#include <EquationSystem.h>
#include <ngp_utils/NgpFieldUtils.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>





namespace sierra{
namespace nalu{

MomentumABLWallFuncMaskUtil::MomentumABLWallFuncMaskUtil(Realm& realm, stk::mesh::Part* part):
  Algorithm(realm, part),
  maskIndex_(get_field_ordinal(realm.meta_data(), "abl_wall_no_slip_wall_func_mask", stk::topology::EDGE_RANK))
{}

void
MomentumABLWallFuncMaskUtil::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meta = realm_.meta_data();
  const auto ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  
  const auto myMask = fieldMgr.get_field<double>(maskIndex_); // nmatula double check this

  // we need this selector to only include edges that touch the abl wall
  const stk::mesh::Selector sel = (meta.locally_owned_part() | meta.globally_shared_part()) &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const auto& bulk = realm_.bulk_data();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, sel);

  auto* edge_field = meta.get_fields()[maskIndex_];
  // auto* node_field = meta.get_fields()[maskIndexNode_];
  
  for (const auto* ib : buckets) {
    for (auto node : *ib) {
      // *(double*)stk::mesh::field(*node_field, node) = 0;
      const auto* connected_edges = bulk.begin_edges(node);
      ThrowRequire(bulk.num_edges(node) > 2 && bulk.num_edges(node) < 6);
      //std::cerr << "test: edge id: " << bulk.identifier(connected_edges[0])
      //          << " for node: " << bulk.identifier(node) << std::endl;
      


      for (int edge_ord = 0; edge_ord < bulk.num_edges(node); ++edge_ord)
        //*stk::mesh::field_data(*edge_field, connected_edges[edge_ord]) = 0;
        *((double*)stk::mesh::field_data(*edge_field, connected_edges[edge_ord])) = 0;
    }
  }

  //nalu_ngp::run_edge_algorithm(
  //  "init abl mask", ngpMesh, sel,
  //  KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
  //    const auto edge = eInfo.meshIdx;
  //    //myMask.get(0, edge) = 0.0;
  //    myMask.get(edge, 0) = 0.0; // nmatula
  //  });


}

}
}