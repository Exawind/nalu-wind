// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPFIELDUTILS_H
#define NGPFIELDUTILS_H

#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"

#include <array>

namespace sierra {
namespace nalu {
namespace nalu_ngp {

template <
  typename T = double,
  typename Mesh = stk::mesh::NgpMesh,
  typename FieldManager = nalu_ngp::FieldManager>
inline stk::mesh::NgpField<T>&
get_ngp_field(
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const std::string& fieldName,
  const stk::mesh::EntityRank& rank = stk::topology::NODE_RANK)
{
  return meshInfo.ngp_field_manager().template get_field<T>(
    get_field_ordinal(meshInfo.meta(), fieldName, rank));
}

template <
  typename T = double,
  typename Mesh = stk::mesh::NgpMesh,
  typename FieldManager = nalu_ngp::FieldManager>
inline stk::mesh::NgpField<T>&
get_ngp_field(
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const std::string& fieldName,
  const stk::mesh::FieldState state,
  const stk::mesh::EntityRank& rank = stk::topology::NODE_RANK)
{
  return meshInfo.ngp_field_manager().template get_field<T>(
    get_field_ordinal(meshInfo.meta(), fieldName, state, rank));
}

} // namespace nalu_ngp
} // namespace nalu
} // namespace sierra

#endif /* NGPFIELDUTILS_H */
