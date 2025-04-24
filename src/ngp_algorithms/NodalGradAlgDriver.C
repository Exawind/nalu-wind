// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "Realm.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

template <typename GradPhiType>
NodalGradAlgDriver<GradPhiType>::NodalGradAlgDriver(
  Realm& realm, const std::string& phiName, const std::string& gradPhiName)
  : NgpAlgDriver(realm), phiName_(phiName), gradPhiName_(gradPhiName)
{
}

template <typename GradPhiType>
void
NodalGradAlgDriver<GradPhiType>::pre_work()
{
  auto grad_phi = nalu_ngp::get_ngp_field(realm_.mesh_info(), gradPhiName_);
  grad_phi.set_all(stk::mesh::get_updated_ngp_mesh(realm_.bulk_data()), 0.0);
}

template <typename GradPhiType>
void
NodalGradAlgDriver<GradPhiType>::post_work()
{
  // TODO: Revisit logic after STK updates to ngp parallel updates
  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  const auto& meshInfo = realm_.mesh_info();

  auto* phi =
    meta.template get_field<double>(stk::topology::NODE_RANK, phiName_);

  auto* gradPhi =
    meta.template get_field<double>(stk::topology::NODE_RANK, gradPhiName_);
  auto& ngpGradPhi = nalu_ngp::get_ngp_field(meshInfo, gradPhiName_);
  ngpGradPhi.sync_to_host();

  // we need the overset update for diffusion terms and grad p
  realm_.scatter_sum_with_overset({gradPhi});
}

template class NodalGradAlgDriver<VectorFieldType>;

} // namespace nalu
} // namespace sierra
