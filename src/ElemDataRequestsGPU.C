// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ElemDataRequestsGPU.h"

namespace sierra {
namespace nalu {

ElemDataRequestsGPU::ElemDataRequestsGPU(
  const nalu_ngp::FieldManager& fieldMgr,
  const ElemDataRequests& dataReq,
  unsigned totalFields)
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

  fill_host_fields(dataReq, fieldMgr);
  fill_host_coords_fields(dataReq, fieldMgr);

  copy_to_device();
}

void
ElemDataRequestsGPU::copy_to_device()
{
  if (hostDataEnums[CURRENT_COORDINATES].size() > 0) {
    Kokkos::deep_copy(
      dataEnums[CURRENT_COORDINATES], hostDataEnums[CURRENT_COORDINATES]);
  }
  if (hostDataEnums[MODEL_COORDINATES].size() > 0) {
    Kokkos::deep_copy(
      dataEnums[MODEL_COORDINATES], hostDataEnums[MODEL_COORDINATES]);
  }
  Kokkos::deep_copy(coordsFields_, hostCoordsFields_);
  Kokkos::deep_copy(coordsFieldsTypes_, hostCoordsFieldsTypes_);
  Kokkos::deep_copy(fields, hostFields);
}

void
ElemDataRequestsGPU::fill_host_data_enums(
  const ElemDataRequests& dataReq, COORDS_TYPES ctype)
{
  if (dataReq.get_data_enums(ctype).size() > 0) {
    dataEnums[ctype] = DataEnumView(
      "DataEnumsCurrentCoords", dataReq.get_data_enums(ctype).size());
    hostDataEnums[ctype] = Kokkos::create_mirror_view(dataEnums[ctype]);
    unsigned i = 0;
    for (ELEM_DATA_NEEDED d : dataReq.get_data_enums(ctype)) {
      hostDataEnums[ctype](i++) = d;
    }
  }
}

void
ElemDataRequestsGPU::fill_host_fields(
  const ElemDataRequests& dataReq, const nalu_ngp::FieldManager& fieldMgr)
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
    hostFields(i++) = FieldInfoType(
      fieldMgr.get_field<double>(finfo.field->mesh_meta_data_ordinal()),
      finfo.scalarsDim1, finfo.scalarsDim2);
  }
}

void
ElemDataRequestsGPU::fill_host_coords_fields(
  const ElemDataRequests& dataReq, const nalu_ngp::FieldManager& fieldMgr)
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
    hostCoordsFields_(i) = CoordFieldInfo(
      fieldMgr.get_field<double>(iter.second->mesh_meta_data_ordinal()));
    hostCoordsFieldsTypes_(i) = iter.first;
    ++i;
  }
}

} // namespace nalu
} // namespace sierra
