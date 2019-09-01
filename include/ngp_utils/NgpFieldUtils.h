/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPFIELDUTILS_H
#define NGPFIELDUTILS_H

#include "ngp_utils/NgpTypes.h"

namespace sierra {
namespace nalu {

namespace nalu_ngp {

template <typename Mesh, typename FieldManager, typename DataType = double>
inline void
copy_field_to_device(
  const MeshInfo<Mesh, FieldManager>& meshInfo, const stk::mesh::FieldBase& field)
{
  const auto fieldID = field.mesh_meta_data_ordinal();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpFld = fieldMgr.template get_field<DataType>(fieldID);

  ngpFld.modify_on_host();
  ngpFld.sync_to_device();
}

template <typename Mesh, typename FieldManager, typename DataType = double>
inline void
copy_field_to_host(
  const MeshInfo<Mesh, FieldManager>& meshInfo, const stk::mesh::FieldBase& field)
{
  const auto fieldID = field.mesh_meta_data_ordinal();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpFld = fieldMgr.template get_field<DataType>(fieldID);

  ngpFld.modify_on_device();
  ngpFld.sync_to_host();
}

} // nalu_ngp
}  // nalu
}  // sierra


#endif /* NGPFIELDUTILS_H */
