// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ElemDataRequestsGPU.h"
#include "master_element/MasterElementRepo.h"

namespace sierra {
namespace nalu {

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

} // namespace nalu

} // namespace sierra
