/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ElemDataRequestsNGP_h
#define ElemDataRequestsNGP_h

#include <KokkosInterface.h>
#include <Kokkos_Core.hpp>
#include <ElemDataRequests.h>

namespace sierra{
namespace nalu{

//Temporary placeholder to allow storing field pointers in
//kokkos views (see FieldView typedef below).
//This won't be necessary when we get stk's ngp::Field because
//those are stored by value (intended to be copied rather than
//referenced through pointers).
struct FieldPtr {
  const stk::mesh::FieldBase* ptr;

  operator const stk::mesh::FieldBase*() const { return ptr; }
};

class ElemDataRequestsNGP
{
public:
  typedef FieldInfo FieldInfoType;
  typedef Kokkos::View<COORDS_TYPES*, Kokkos::LayoutRight, MemSpace> CoordsTypesView;
  typedef Kokkos::View<ELEM_DATA_NEEDED*, Kokkos::LayoutRight, MemSpace> DataEnumView;
  typedef Kokkos::View<FieldPtr*, Kokkos::LayoutRight, MemSpace> FieldView;
  typedef Kokkos::View<FieldInfoType*, Kokkos::LayoutRight, MemSpace> FieldInfoView;

  ElemDataRequestsNGP(const ElemDataRequests& dataReq, unsigned totalFields)
    : totalNumFields(totalFields),
      dataEnums(),
      hostDataEnums(),
      coordsFields_(),
      hostCoordsFields_(),
      coordsFieldsTypes_(),
      hostCoordsFieldsTypes_(),
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

  ~ElemDataRequestsNGP() {}

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

  MasterElement *get_cvfem_face_me() const {return meFC_;}
  MasterElement *get_cvfem_volume_me() const {return meSCV_;}
  MasterElement *get_cvfem_surface_me() const {return meSCS_;}
  MasterElement *get_fem_volume_me() const {return meFEM_;}

  unsigned get_total_num_fields() const { return totalNumFields; }

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
    for(const FieldInfoType& finfo : dataReq.get_fields()) {
      hostFields(i++) = finfo;
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
      hostCoordsFields_(i) = {iter.second};
      hostCoordsFieldsTypes_(i) = iter.first;
      ++i;
    }
  }

  unsigned totalNumFields;
  DataEnumView dataEnums[MAX_COORDS_TYPES];
  DataEnumView::HostMirror hostDataEnums[MAX_COORDS_TYPES];

  FieldView coordsFields_;
  FieldView::HostMirror hostCoordsFields_;
  CoordsTypesView coordsFieldsTypes_;
  CoordsTypesView::HostMirror hostCoordsFieldsTypes_;

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

