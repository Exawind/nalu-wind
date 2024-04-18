// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ElemDataRequests.h>

namespace sierra {
namespace nalu {

void
ElemDataRequests::add_gathered_nodal_field(
  const stk::mesh::FieldBase& field, unsigned scalarsPerNode)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::NODE_RANK,
    "ElemDataRequests ERROR, add_gathered_nodal_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==NODE_RANK");

  FieldInfo fieldInfo(&field, scalarsPerNode);
  FieldSet::iterator iter = fields.find(fieldInfo);
  if (iter == fields.end()) {
    fields.insert(fieldInfo);
  } else {
    STK_ThrowRequireMsg(
      iter->scalarsDim1 == scalarsPerNode,
      "ElemDataRequests ERROR, gathered-nodal-field "
        << field.name() << " requested with scalarsPerNode==" << scalarsPerNode
        << ", but previously requested with scalarsPerNode=="
        << iter->scalarsDim1);
  }
}

void
ElemDataRequests::add_gathered_nodal_field(
  const stk::mesh::FieldBase& field, unsigned tensorDim1, unsigned tensorDim2)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::NODE_RANK,
    "ElemDataRequests ERROR, add_gathered_nodal_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==NODE_RANK");

  FieldInfo fieldInfo(&field, tensorDim1, tensorDim2);
  FieldSet::iterator iter = fields.find(fieldInfo);
  if (iter == fields.end()) {
    fields.insert(fieldInfo);
  } else {
    STK_ThrowRequireMsg(
      iter->scalarsDim1 == tensorDim1 && iter->scalarsDim2 == tensorDim2,
      "ElemDataRequests ERROR, gathered-nodal-field "
        << field.name() << " requested with tensorDim1==" << tensorDim1
        << ",tensorDim2==" << tensorDim2
        << ", but previously requested with tensorDim1==" << iter->scalarsDim1
        << ",tensorDim2==" << iter->scalarsDim2);
  }
}

void
ElemDataRequests::add_ip_field(
  const stk::mesh::FieldBase& field, unsigned scalarsPerElement)
{
  FieldInfo fieldInfo(&field, scalarsPerElement);
  FieldSet::iterator iter = fields.find(fieldInfo);
  if (iter == fields.end()) {
    fields.insert(fieldInfo);
  } else {
    STK_ThrowRequireMsg(
      iter->scalarsDim1 == scalarsPerElement,
      "ElemDataRequests ERROR, ip-field "
        << field.name()
        << " requested with scalarsPerElement==" << scalarsPerElement
        << ", but previously requested with scalarsPerElement=="
        << iter->scalarsDim1);
  }
}

void
ElemDataRequests::add_face_field(
  const stk::mesh::FieldBase& field, unsigned scalarsPerFace)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::FACE_RANK ||
      field.entity_rank() == stk::topology::EDGE_RANK,
    "ElemDataRequests ERROR, add_face_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==ELEM_RANK");

  add_ip_field(field, scalarsPerFace);
}

void
ElemDataRequests::add_element_field(
  const stk::mesh::FieldBase& field, unsigned scalarsPerElement)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::ELEM_RANK,
    "ElemDataRequests ERROR, add_element_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==ELEM_RANK");

  add_ip_field(field, scalarsPerElement);
}

void
ElemDataRequests::add_ip_field(
  const stk::mesh::FieldBase& field, unsigned tensorDim1, unsigned tensorDim2)
{
  FieldInfo fieldInfo(&field, tensorDim1, tensorDim2);
  FieldSet::iterator iter = fields.find(fieldInfo);
  if (iter == fields.end()) {
    fields.insert(fieldInfo);
  } else {
    STK_ThrowRequireMsg(
      iter->scalarsDim1 == tensorDim1 && iter->scalarsDim2 == tensorDim2,
      "ElemDataRequests ERROR, ip-field "
        << field.name() << " requested with tensorDim1==" << tensorDim1
        << ",tensorDim2==" << tensorDim2
        << ", but previously requested with tensorDim1==" << iter->scalarsDim1
        << ",tensorDim2==" << iter->scalarsDim2);
  }
}

void
ElemDataRequests::add_face_field(
  const stk::mesh::FieldBase& field, unsigned tensorDim1, unsigned tensorDim2)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::FACE_RANK ||
      field.entity_rank() == stk::topology::EDGE_RANK,
    "ElemDataRequests ERROR, add_face_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==ELEM_RANK");

  add_ip_field(field, tensorDim1, tensorDim2);
}

void
ElemDataRequests::add_element_field(
  const stk::mesh::FieldBase& field, unsigned tensorDim1, unsigned tensorDim2)
{
  STK_ThrowRequireMsg(
    field.entity_rank() == stk::topology::ELEM_RANK,
    "ElemDataRequests ERROR, add_element_field called with field "
      << field.name() << " which has entity-rank==" << field.entity_rank()
      << " but is expected to have rank==ELEM_RANK");

  add_ip_field(field, tensorDim1, tensorDim2);
}

void
ElemDataRequests::add_coordinates_field(
  const stk::mesh::FieldBase& field,
  unsigned scalarsPerNode,
  COORDS_TYPES cType)
{
  coordsFields_[cType] = &field;
  add_gathered_nodal_field(field, scalarsPerNode);
}

void
ElemDataRequests::add_master_element_call(
  ELEM_DATA_NEEDED data, COORDS_TYPES cType)
{
  auto it = coordsFields_.find(cType);
  STK_NGP_ThrowRequireMsg(
    it != coordsFields_.end(),
    "ElemDataRequests:add_master_element_call: Coordinates field "
    "must be registered to ElemDataRequests before registering MasterElement "
    "call");

  // Check that the appropriate MasterElement has been registered
  if ((data >= BEGIN_FC) && (data <= END_FC)) {
    STK_NGP_ThrowRequire(meFC_ != nullptr);
  } else if ((data >= BEGIN_SCS) && (data <= END_SCS)) {
    STK_NGP_ThrowRequire(meSCS_ != nullptr);
  } else if ((data >= BEGIN_SCV) && (data <= END_SCV)) {
    STK_NGP_ThrowRequire(meSCV_ != nullptr);
  } else if ((data >= BEGIN_FEM) && (data <= END_FEM)) {
    STK_NGP_ThrowRequire(meFEM_ != nullptr);
  }

  dataEnums[cType].insert(data);
}

} // namespace nalu
} // namespace sierra
