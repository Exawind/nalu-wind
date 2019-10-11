/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPFIELDUTILS_H
#define NGPFIELDUTILS_H

#include "ngp_utils/NgpTypes.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/FieldBase.hpp"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

template <
  typename T = double,
  typename Mesh = ngp::Mesh,
  typename FieldManager = ngp::FieldManager>
inline ngp::Field<T>& get_ngp_field(
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const std::string& fieldName,
  const stk::mesh::EntityRank& rank = stk::topology::NODE_RANK)
{
  return meshInfo.ngp_field_manager().template get_field<T>(
    get_field_ordinal(meshInfo.meta(), fieldName, rank));
}

template <
  typename T = double,
  typename Mesh = ngp::Mesh,
  typename FieldManager = ngp::FieldManager>
inline ngp::Field<T>& get_ngp_field(
  const MeshInfo<Mesh, FieldManager>& meshInfo,
  const std::string& fieldName,
  const stk::mesh::FieldState state,
  const stk::mesh::EntityRank& rank = stk::topology::NODE_RANK)
{
  return meshInfo.ngp_field_manager().template get_field<T>(
    get_field_ordinal(meshInfo.meta(), fieldName, state, rank));
}

}
}  // nalu
}  // sierra


#endif /* NGPFIELDUTILS_H */
