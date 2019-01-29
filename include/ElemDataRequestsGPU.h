/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ElemDataRequestsGPU_h
#define ElemDataRequestsGPU_h

#include <KokkosInterface.h>
#include <Kokkos_Core.hpp>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <stk_ngp/Ngp.hpp>

namespace sierra{
namespace nalu{

struct FieldInfoNGP {
  FieldInfoNGP(const stk::mesh::FieldBase* fld, unsigned scalars)
  : field(fld->get_mesh(), *fld), scalarsDim1(scalars), scalarsDim2(0)
  {}  
  FieldInfoNGP(const stk::mesh::FieldBase* fld, unsigned tensorDim1, unsigned tensorDim2)
  : field(fld->get_mesh(), *fld), scalarsDim1(tensorDim1), scalarsDim2(tensorDim2)
  {}  
  KOKKOS_FUNCTION
  FieldInfoNGP(const FieldInfoNGP& rhs)
  : field(rhs.field), scalarsDim1(rhs.scalarsDim1), scalarsDim2(rhs.scalarsDim2)
  {}
  KOKKOS_FUNCTION
  FieldInfoNGP()
  : field(), scalarsDim1(0), scalarsDim2(0)
  {}

  NGPDoubleFieldType field;
  unsigned scalarsDim1;
  unsigned scalarsDim2;
};

KOKKOS_INLINE_FUNCTION
stk::mesh::EntityRank get_entity_rank(const FieldInfoNGP& fieldInfo)
{
  return fieldInfo.field.get_rank();
}

KOKKOS_INLINE_FUNCTION
unsigned get_field_ordinal(const FieldInfoNGP& fieldInfo)
{
  return fieldInfo.field.get_ordinal();
}

class ElemDataRequestsGPU
{
public:
  typedef FieldInfoNGP FieldInfoType;
  typedef Kokkos::View<COORDS_TYPES*, Kokkos::LayoutRight, MemSpace> CoordsTypesView;
  typedef Kokkos::View<ELEM_DATA_NEEDED*, Kokkos::LayoutRight, MemSpace> DataEnumView;
  typedef Kokkos::View<NGPDoubleFieldType*, Kokkos::LayoutRight, MemSpace> FieldView;
  typedef Kokkos::View<FieldInfoType*, Kokkos::LayoutRight, MemSpace> FieldInfoView;

  ElemDataRequestsGPU(const ElemDataRequests& dataReq, unsigned totalFields)
    : dataEnums(),
      hostDataEnums(),
      coordsFields_(),
      hostCoordsFields_(),
      coordsFieldsTypes_(),
      hostCoordsFieldsTypes_(),
      totalNumFields(totalFields),
      fields(),
      hostFields(),
      meFC_(dataReq.get_cvfem_face_me()),
      meSCS_(dataReq.get_cvfem_surface_me()),
      meSCV_(dataReq.get_cvfem_volume_me()),
      meFEM_(dataReq.get_fem_volume_me())
  {
    fill_host_data_enums(dataReq, CURRENT_COORDINATES);
    fill_host_data_enums(dataReq, MODEL_COORDINATES);

    fill_host_fields(dataReq);
    fill_host_coords_fields(dataReq);

    copy_to_device();
  }

  ~ElemDataRequestsGPU() {}

  void add_cvfem_face_me(MasterElement *meFC)
  { meFC_ = meFC; }

  void add_cvfem_volume_me(MasterElement *meSCV)
  { meSCV_ = meSCV; }

  void add_cvfem_surface_me(MasterElement *meSCS)
  { meSCS_ = meSCS; }

  void add_fem_volume_me(MasterElement *meFEM)
  { meFEM_ = meFEM; }

  KOKKOS_FUNCTION
  const DataEnumView& get_data_enums(const COORDS_TYPES cType) const
  { return dataEnums[cType]; }

  KOKKOS_FUNCTION
  const FieldView& get_coordinates_fields() const
  { return coordsFields_; }

  KOKKOS_FUNCTION
  const CoordsTypesView& get_coordinates_types() const
  { return coordsFieldsTypes_; }

  KOKKOS_FUNCTION
  const FieldInfoView& get_fields() const { return fields; }  

  KOKKOS_FUNCTION
  unsigned get_total_num_fields() const { return totalNumFields; }

  KOKKOS_FUNCTION
  MasterElement *get_cvfem_face_me() const {return meFC_;}
  KOKKOS_FUNCTION
  MasterElement *get_cvfem_volume_me() const {return meSCV_;}
  KOKKOS_FUNCTION
  MasterElement *get_cvfem_surface_me() const {return meSCS_;}
  KOKKOS_FUNCTION
  MasterElement *get_fem_volume_me() const {return meFEM_;}

private:
  void copy_to_device()
  {
    if (hostDataEnums[CURRENT_COORDINATES].size() > 0) {
      Kokkos::deep_copy(dataEnums[CURRENT_COORDINATES], hostDataEnums[CURRENT_COORDINATES]);
    }
    if (hostDataEnums[MODEL_COORDINATES].size() > 0) {
      Kokkos::deep_copy(dataEnums[MODEL_COORDINATES], hostDataEnums[MODEL_COORDINATES]);
    }
    Kokkos::deep_copy(coordsFields_, hostCoordsFields_);
    Kokkos::deep_copy(coordsFieldsTypes_, hostCoordsFieldsTypes_);
    Kokkos::deep_copy(fields, hostFields);
  }

  void fill_host_data_enums(const ElemDataRequests& dataReq, COORDS_TYPES ctype)
  {
    if (dataReq.get_data_enums(ctype).size() > 0) {
      dataEnums[ctype] = DataEnumView("DataEnumsCurrentCoords", dataReq.get_data_enums(ctype).size());
      hostDataEnums[ctype] = Kokkos::create_mirror_view(dataEnums[ctype]);
      unsigned i=0;
      for(ELEM_DATA_NEEDED d : dataReq.get_data_enums(ctype)) {
        hostDataEnums[ctype](i++) = d;
      }
    }
  }

  void fill_host_fields(const ElemDataRequests& dataReq)
  {
    fields = FieldInfoView("Fields", dataReq.get_fields().size());
    hostFields = Kokkos::create_mirror_view(fields);
    unsigned i = 0;
    for(const FieldInfo& finfo : dataReq.get_fields()) {
      hostFields(i++) = FieldInfoType(finfo.field, finfo.scalarsDim1, finfo.scalarsDim2);
    }
  }
 
  void fill_host_coords_fields(const ElemDataRequests& dataReq)
  {
    coordsFields_ = FieldView("CoordsFields", dataReq.get_coordinates_map().size());
    coordsFieldsTypes_ = CoordsTypesView("CoordsFieldsTypes", dataReq.get_coordinates_map().size());

    hostCoordsFields_ = Kokkos::create_mirror_view(coordsFields_);
    hostCoordsFieldsTypes_ = Kokkos::create_mirror_view(coordsFieldsTypes_);

    unsigned i = 0;
    for(auto iter : dataReq.get_coordinates_map()) {
      hostCoordsFields_(i) = NGPDoubleFieldType(iter.second->get_mesh(), *iter.second);
      hostCoordsFieldsTypes_(i) = iter.first;
      ++i;
    }
  }

  DataEnumView dataEnums[MAX_COORDS_TYPES];
  DataEnumView::HostMirror hostDataEnums[MAX_COORDS_TYPES];

  FieldView coordsFields_;
  FieldView::HostMirror hostCoordsFields_;
  CoordsTypesView coordsFieldsTypes_;
  CoordsTypesView::HostMirror hostCoordsFieldsTypes_;

  unsigned totalNumFields;
  FieldInfoView fields;
  FieldInfoView::HostMirror hostFields;

  MasterElement *meFC_;
  MasterElement *meSCS_;
  MasterElement *meSCV_;
  MasterElement *meFEM_;
};

} // namespace nalu
} // namespace Sierra

#endif

