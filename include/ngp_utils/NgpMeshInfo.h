// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef NGPMESHINFO_H
#define NGPMESHINFO_H

/** \file
 *  \brief NGP mesh information objects
 */

#include "stk_ngp/Ngp.hpp"
#include "stk_ngp/NgpFieldManager.hpp"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** STK mesh object holder
 *
 *  This lightweight class carries information regarding the STK meshes both the
 *  non-NGP versions (MetaData/BulkData) as well as the `ngp::Mesh` and
 *  `ngp::FieldManager` instances.
 */
template <typename Mesh = ngp::Mesh, typename FieldManager = ngp::FieldManager>
class MeshInfo
{
public:
  using NgpMeshType = Mesh;
  using NgpFieldManagerType = FieldManager;

  MeshInfo(
    const stk::mesh::BulkData& bulk
  ) : bulk_(bulk),
      meta_(bulk.mesh_meta_data()),
      ngpMesh_(bulk),
      ngpFieldMgr_(bulk)
  {}

  ~MeshInfo() = default;

  MeshInfo() = delete;
  MeshInfo(const MeshInfo&) = delete;
  MeshInfo& operator=(const MeshInfo&) = delete;

  inline const stk::mesh::BulkData& bulk() const { return bulk_; }

  inline const stk::mesh::MetaData& meta() const { return meta_; }

  inline const Mesh& ngp_mesh() const { return ngpMesh_; }

  inline const FieldManager& ngp_field_manager() const { return ngpFieldMgr_; }

  inline int ndim() const { return meta_.spatial_dimension(); }

  inline size_t num_fields() const { return meta_.get_fields().size(); }

private:
  //! Reference to the bulk data for the STK mesh
  const stk::mesh::BulkData& bulk_;

  //! Reference to the mesh meta data
  const stk::mesh::MetaData& meta_;

  //! NGP mesh instance
  const Mesh ngpMesh_;

  //! NGP field manager instance
  const FieldManager ngpFieldMgr_;
};

}  // nalu_ngp
}  // nalu
}  // sierra


#endif /* NGPMESHINFO_H */
