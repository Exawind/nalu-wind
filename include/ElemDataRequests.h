// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ElemDataRequests_h
#define ElemDataRequests_h

#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Ngp.hpp>

#include <set>
#include <array>
#include <map>

namespace sierra {
namespace nalu {

class MasterElement;

enum ELEM_DATA_NEEDED {
  FC_AREAV = 0,
  BEGIN_FC = FC_AREAV,
  FC_SHAPE_FCN,
  FC_SHIFTED_SHAPE_FCN,
  END_FC = FC_SHIFTED_SHAPE_FCN,
  SCS_AREAV,
  BEGIN_SCS = SCS_AREAV,
  SCS_FACE_GRAD_OP,
  SCS_SHIFTED_FACE_GRAD_OP,
  SCS_GRAD_OP,
  SCS_SHIFTED_GRAD_OP,
  SCS_GIJ,
  SCS_MIJ,
  SCS_SHAPE_FCN,
  SCS_SHIFTED_SHAPE_FCN,
  END_SCS = SCS_SHIFTED_SHAPE_FCN,
  SCV_VOLUME,
  BEGIN_SCV = SCV_VOLUME,
  SCV_MIJ,
  SCV_GRAD_OP,
  SCV_SHIFTED_GRAD_OP,
  SCV_SHAPE_FCN,
  SCV_SHIFTED_SHAPE_FCN,
  END_SCV = SCV_SHIFTED_SHAPE_FCN,
  FEM_GRAD_OP,
  BEGIN_FEM = FEM_GRAD_OP,
  FEM_SHIFTED_GRAD_OP,
  FEM_SHAPE_FCN,
  FEM_SHIFTED_SHAPE_FCN,
  END_FEM = FEM_SHIFTED_SHAPE_FCN
};

enum COORDS_TYPES {
  CURRENT_COORDINATES = 0,
  MODEL_COORDINATES,
  MAX_COORDS_TYPES,
};

static const std::string CoordinatesTypeNames[] = {"current_coordinates",
                                                   "model_coordinates"};

struct FieldInfo
{
  FieldInfo(const stk::mesh::FieldBase* fld, unsigned scalars)
    : field(fld), scalarsDim1(scalars), scalarsDim2(0)
  {
  }
  FieldInfo(
    const stk::mesh::FieldBase* fld, unsigned tensorDim1, unsigned tensorDim2)
    : field(fld), scalarsDim1(tensorDim1), scalarsDim2(tensorDim2)
  {
  }
  FieldInfo() : field(nullptr), scalarsDim1(0), scalarsDim2(0) {}

  const stk::mesh::FieldBase* field;
  unsigned scalarsDim1;
  unsigned scalarsDim2;
};

inline stk::mesh::EntityRank
get_entity_rank(const FieldInfo& fieldInfo)
{
  return fieldInfo.field->entity_rank();
}

inline unsigned
get_field_ordinal(const FieldInfo& fieldInfo)
{
  return fieldInfo.field->mesh_meta_data_ordinal();
}

struct FieldInfoLess
{
  bool operator()(const FieldInfo& lhs, const FieldInfo& rhs) const
  {
    return lhs.field->mesh_meta_data_ordinal() <
           rhs.field->mesh_meta_data_ordinal();
  }
};

typedef std::set<FieldInfo, FieldInfoLess> FieldSet;

class ElemDataRequests
{
public:
  ElemDataRequests(const stk::mesh::MetaData& meta)
    : meta_(meta),
      dataEnums(),
      coordsFields_(),
      fields(),
      meFC_(nullptr),
      meSCS_(nullptr),
      meSCV_(nullptr),
      meFEM_(nullptr)
  {
  }

  void add_master_element_call(
    ELEM_DATA_NEEDED data, COORDS_TYPES cType = CURRENT_COORDINATES);

  void add_gathered_nodal_field(
    const stk::mesh::FieldBase& field, unsigned scalarsPerNode);

  void add_gathered_nodal_field(
    const stk::mesh::FieldBase& field,
    unsigned tensorDim1,
    unsigned tensorDim2);

  void
  add_face_field(const stk::mesh::FieldBase& field, unsigned scalarsPerFace);
  void
  add_ip_field(const stk::mesh::FieldBase& field, unsigned scalarsPerElement);
  void add_element_field(
    const stk::mesh::FieldBase& field, unsigned scalarsPerElement);

  void add_face_field(
    const stk::mesh::FieldBase& field,
    unsigned tensorDim1,
    unsigned tensorDim2);
  void add_ip_field(
    const stk::mesh::FieldBase& field,
    unsigned tensorDim1,
    unsigned tensorDim2);
  void add_element_field(
    const stk::mesh::FieldBase& field,
    unsigned tensorDim1,
    unsigned tensorDim2);

  inline void add_gathered_nodal_field(unsigned field, unsigned scalarsPerNode)
  {
    add_gathered_nodal_field(*meta_.get_fields()[field], scalarsPerNode);
  }

  inline void add_gathered_nodal_field(
    unsigned field, unsigned tensorDim1, unsigned tensorDim2)
  {
    add_gathered_nodal_field(
      *meta_.get_fields()[field], tensorDim1, tensorDim2);
  }

  inline void add_face_field(unsigned field, unsigned scalarsPerFace)
  {
    add_face_field(*meta_.get_fields()[field], scalarsPerFace);
  }

  inline void add_element_field(unsigned field, unsigned scalarsPerElement)
  {
    add_element_field(*meta_.get_fields()[field], scalarsPerElement);
  }

  inline void
  add_face_field(unsigned field, unsigned tensorDim1, unsigned tensorDim2)
  {
    add_face_field(*meta_.get_fields()[field], tensorDim1, tensorDim2);
  }

  inline void
  add_element_field(unsigned field, unsigned tensorDim1, unsigned tensorDim2)
  {
    add_element_field(*meta_.get_fields()[field], tensorDim1, tensorDim2);
  }

  void add_coordinates_field(
    const stk::mesh::FieldBase& field,
    unsigned scalarsPerNode,
    COORDS_TYPES cType);

  inline void add_coordinates_field(
    unsigned field, unsigned scalarsPerNode, COORDS_TYPES cType)
  {
    add_coordinates_field(*meta_.get_fields()[field], scalarsPerNode, cType);
  }

  void add_cvfem_face_me(MasterElement* meFC) { meFC_ = meFC; }

  void add_cvfem_volume_me(MasterElement* meSCV) { meSCV_ = meSCV; }

  void add_cvfem_surface_me(MasterElement* meSCS) { meSCS_ = meSCS; }

  void add_fem_volume_me(MasterElement* meFEM) { meFEM_ = meFEM; }

  const std::set<ELEM_DATA_NEEDED>&
  get_data_enums(const COORDS_TYPES cType) const
  {
    return dataEnums[cType];
  }

  const stk::mesh::FieldBase*
  get_coordinates_field(const COORDS_TYPES cType) const
  {
    auto it = coordsFields_.find(cType);
    NGP_ThrowRequireMsg(
      it != coordsFields_.end(),
      "ElemDataRequests:get_coordinates_field: Coordinates field "
      "must be registered to ElemDataRequests before access");
    return it->second;
  }

  const std::map<COORDS_TYPES, const stk::mesh::FieldBase*>&
  get_coordinates_map() const
  {
    return coordsFields_;
  }

  const FieldSet& get_fields() const { return fields; }
  MasterElement* get_cvfem_face_me() const { return meFC_; }
  MasterElement* get_cvfem_volume_me() const { return meSCV_; }
  MasterElement* get_cvfem_surface_me() const { return meSCS_; }
  MasterElement* get_fem_volume_me() const { return meFEM_; }

private:
  const stk::mesh::MetaData& meta_;
  std::array<std::set<ELEM_DATA_NEEDED>, MAX_COORDS_TYPES> dataEnums;
  std::map<COORDS_TYPES, const stk::mesh::FieldBase*> coordsFields_;
  FieldSet fields;
  MasterElement* meFC_;
  MasterElement* meSCS_;
  MasterElement* meSCV_;
  MasterElement* meFEM_;
};

} // namespace nalu
} // namespace sierra

#endif
