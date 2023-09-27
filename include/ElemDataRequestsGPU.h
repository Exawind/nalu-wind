// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ElemDataRequestsGPU_h
#define ElemDataRequestsGPU_h

#include <KokkosInterface.h>
#include <Kokkos_Core.hpp>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <ngp_utils/NgpFieldManager.h>
#include <FieldManager.h>
#include "master_element/MasterElementRepo.h"

namespace sierra {
namespace nalu {

struct FieldInfoNGP
{
  FieldInfoNGP(const stk::mesh::FieldBase* fld, unsigned scalars)
    : field(stk::mesh::get_updated_ngp_field<double>(*fld)),
      scalarsDim1(scalars),
      scalarsDim2(0)
  {
    field.sync_to_device();
  }
  FieldInfoNGP(
    const stk::mesh::FieldBase* fld, unsigned tensorDim1, unsigned tensorDim2)
    : field(stk::mesh::get_updated_ngp_field<double>(*fld)),
      scalarsDim1(tensorDim1),
      scalarsDim2(tensorDim2)
  {
    field.sync_to_device();
  }
  FieldInfoNGP(NGPDoubleFieldType& fld, unsigned scalars)
    : field(fld), scalarsDim1(scalars), scalarsDim2(0)
  {
  }
  FieldInfoNGP(
    NGPDoubleFieldType& fld, unsigned tensorDim1, unsigned tensorDim2)
    : field(fld), scalarsDim1(tensorDim1), scalarsDim2(tensorDim2)
  {
  }
  KOKKOS_FUNCTION
  FieldInfoNGP(const FieldInfoNGP& rhs)
    : field(rhs.field),
      scalarsDim1(rhs.scalarsDim1),
      scalarsDim2(rhs.scalarsDim2)
  {
  }
  KOKKOS_FUNCTION
  FieldInfoNGP() : field(), scalarsDim1(0), scalarsDim2(0) {}

  KOKKOS_DEFAULTED_FUNCTION
  FieldInfoNGP& operator=(const FieldInfoNGP&) = default;

  NGPDoubleFieldType field;
  unsigned scalarsDim1;
  unsigned scalarsDim2;
};

struct CoordFieldInfo
{
  CoordFieldInfo(NGPDoubleFieldType& fld) : coordField(fld) {}

  KOKKOS_DEFAULTED_FUNCTION
  CoordFieldInfo() = default;

  KOKKOS_DEFAULTED_FUNCTION
  CoordFieldInfo(const CoordFieldInfo&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  ~CoordFieldInfo() = default;

  KOKKOS_FUNCTION
  operator const NGPDoubleFieldType&() const { return coordField; }

  NGPDoubleFieldType coordField;
};

KOKKOS_INLINE_FUNCTION
stk::mesh::EntityRank
get_entity_rank(const FieldInfoNGP& fieldInfo)
{
  return fieldInfo.field.get_rank();
}

KOKKOS_INLINE_FUNCTION
unsigned
get_field_ordinal(const FieldInfoNGP& fieldInfo)
{
  return fieldInfo.field.get_ordinal();
}

KOKKOS_INLINE_FUNCTION
unsigned
get_field_ordinal(const NGPDoubleFieldType& field)
{
  return field.get_ordinal();
}

class ElemDataRequestsGPU
{
public:
  typedef FieldInfoNGP FieldInfoType;
  typedef NGPDoubleFieldType FieldType;
  typedef Kokkos::View<COORDS_TYPES*, Kokkos::LayoutRight, MemSpace>
    CoordsTypesView;
  typedef Kokkos::View<ELEM_DATA_NEEDED*, Kokkos::LayoutRight, MemSpace>
    DataEnumView;
  typedef Kokkos::View<CoordFieldInfo*, Kokkos::LayoutRight, MemSpace>
    FieldView;
  typedef Kokkos::View<FieldInfoType*, Kokkos::LayoutRight, MemSpace>
    FieldInfoView;

  template <typename T>
  ElemDataRequestsGPU(const T& fieldMgr, const ElemDataRequests& dataReq);

  KOKKOS_FUNCTION ~ElemDataRequestsGPU() {}

  void add_cvfem_face_me(MasterElement* meFC) { meFC_ = meFC; }

  void add_cvfem_volume_me(MasterElement* meSCV) { meSCV_ = meSCV; }

  void add_cvfem_surface_me(MasterElement* meSCS) { meSCS_ = meSCS; }

  void add_fem_volume_me(MasterElement* meFEM) { meFEM_ = meFEM; }

  KOKKOS_FUNCTION
  const DataEnumView& get_data_enums(const COORDS_TYPES cType) const
  {
    return dataEnums[cType];
  }

  KOKKOS_FUNCTION
  const FieldView& get_coordinates_fields() const { return coordsFields_; }

  KOKKOS_FUNCTION
  const CoordsTypesView& get_coordinates_types() const
  {
    return coordsFieldsTypes_;
  }

  KOKKOS_FUNCTION
  const FieldInfoView& get_fields() const { return fields; }

  const DataEnumView::HostMirror&
  get_host_data_enums(const COORDS_TYPES cType) const
  {
    return hostDataEnums[cType];
  }

  const FieldView::HostMirror& get_host_coordinates_fields() const
  {
    return hostCoordsFields_;
  }

  const CoordsTypesView::HostMirror& get_host_coordinates_types() const
  {
    return hostCoordsFieldsTypes_;
  }

  const FieldInfoView::HostMirror& get_host_fields() const
  {
    return hostFields;
  }

  KOKKOS_FUNCTION
  unsigned get_total_num_fields() const { return totalNumFields; }

  KOKKOS_FUNCTION
  MasterElement* get_cvfem_face_me() const { return meFC_; }
  KOKKOS_FUNCTION
  MasterElement* get_cvfem_volume_me() const { return meSCV_; }
  KOKKOS_FUNCTION
  MasterElement* get_cvfem_surface_me() const { return meSCS_; }
  KOKKOS_FUNCTION
  MasterElement* get_fem_volume_me() const { return meFEM_; }

private:
  void copy_to_device();

  void
  fill_host_data_enums(const ElemDataRequests& dataReq, COORDS_TYPES ctype);

  template <typename T>
  void fill_host_fields(const ElemDataRequests& dataReq, const T& fieldMgr);

  template <typename T, typename U>
  auto& get_coord_ptr(const T& fieldMgr, const U& iter) const;

  template <typename T>
  stk::mesh::NgpField<double>&
  get_field_ptr(const T& fieldMgr, const FieldInfo& finfo) const;

  template <typename T>
  void
  fill_host_coords_fields(const ElemDataRequests& dataReq, const T& fieldMgr);

  DataEnumView dataEnums[MAX_COORDS_TYPES];
  DataEnumView::HostMirror hostDataEnums[MAX_COORDS_TYPES];

  FieldView coordsFields_;
  FieldView::HostMirror hostCoordsFields_;
  CoordsTypesView coordsFieldsTypes_;
  CoordsTypesView::HostMirror hostCoordsFieldsTypes_;

  unsigned totalNumFields;
  FieldInfoView fields;
  FieldInfoView::HostMirror hostFields;

  MasterElement* meFC_;
  MasterElement* meSCS_;
  MasterElement* meSCV_;
  MasterElement* meFEM_;
};

template <typename T>
inline ElemDataRequestsGPU::ElemDataRequestsGPU(
  const T& fieldMgr, const ElemDataRequests& dataReq)
  : dataEnums(),
    hostDataEnums(),
    coordsFields_(),
    hostCoordsFields_(),
    coordsFieldsTypes_(),
    hostCoordsFieldsTypes_(),
    totalNumFields(fieldMgr.size()),
    fields(),
    hostFields(),
    meFC_(MasterElementRepo::get_surface_dev_ptr_from_host_ptr(
      dataReq.get_cvfem_face_me())),
    meSCS_(MasterElementRepo::get_surface_dev_ptr_from_host_ptr(
      dataReq.get_cvfem_surface_me())),
    meSCV_(MasterElementRepo::get_volume_dev_ptr_from_host_ptr(
      dataReq.get_cvfem_volume_me())),
    meFEM_(MasterElementRepo::get_volume_dev_ptr_from_host_ptr(
      dataReq.get_fem_volume_me()))
{
  fill_host_data_enums(dataReq, CURRENT_COORDINATES);
  fill_host_data_enums(dataReq, MODEL_COORDINATES);

  fill_host_fields(dataReq, fieldMgr);
  fill_host_coords_fields(dataReq, fieldMgr);

  copy_to_device();
}

template <typename T, typename U>
auto&
ElemDataRequestsGPU::get_coord_ptr(const T& fieldMgr, const U& iter) const
{
  if constexpr (std::is_same_v<T, nalu::FieldManager>)
    return fieldMgr.template get_ngp_field_ptr<double>(iter.second->name());
  else
    return fieldMgr.template get_field<double>(
      iter.second->mesh_meta_data_ordinal());
}

template <typename T>
void
ElemDataRequestsGPU::fill_host_coords_fields(
  const ElemDataRequests& dataReq, const T& fieldMgr)
{
#if defined(KOKKOS_ENABLE_GPU)
  coordsFields_ = FieldView(
    Kokkos::ViewAllocateWithoutInitializing("CoordsFields"),
    dataReq.get_coordinates_map().size());
#else
  coordsFields_ =
    FieldView("CoordsFields", dataReq.get_coordinates_map().size());
#endif
  coordsFieldsTypes_ =
    CoordsTypesView("CoordsFieldsTypes", dataReq.get_coordinates_map().size());

  hostCoordsFields_ = Kokkos::create_mirror_view(coordsFields_);
  hostCoordsFieldsTypes_ = Kokkos::create_mirror_view(coordsFieldsTypes_);

  unsigned i = 0;
  for (auto iter : dataReq.get_coordinates_map()) {
    hostCoordsFields_(i) = CoordFieldInfo(get_coord_ptr(fieldMgr, iter));
    hostCoordsFieldsTypes_(i) = iter.first;
    ++i;
  }
}

template <typename T>
stk::mesh::NgpField<double>&
ElemDataRequestsGPU::get_field_ptr(
  const T& fieldMgr, const FieldInfo& finfo) const
{
  if constexpr (std::is_same_v<T, nalu::FieldManager>)
    return fieldMgr.template get_ngp_field_ptr<double>(finfo.field->name());
  else
    return fieldMgr.template get_field<double>(
      finfo.field->mesh_meta_data_ordinal());
}

template <typename T>
void
ElemDataRequestsGPU::fill_host_fields(
  const ElemDataRequests& dataReq, const T& fieldMgr)
{
#if defined(KOKKOS_ENABLE_GPU)
  fields = FieldInfoView(
    Kokkos::ViewAllocateWithoutInitializing("Fields"),
    dataReq.get_fields().size());
#else
  fields = FieldInfoView("Fields", dataReq.get_fields().size());
#endif
  hostFields = Kokkos::create_mirror_view(fields);
  unsigned i = 0;
  for (const FieldInfo& finfo : dataReq.get_fields()) {
    stk::mesh::NgpField<double>& fld_ptr = get_field_ptr(fieldMgr, finfo);
    hostFields(i++) =
      FieldInfoType(fld_ptr, finfo.scalarsDim1, finfo.scalarsDim2);
  }
}
} // namespace nalu
} // namespace sierra

#endif
