// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/UnitTestNgpAlgUtils.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/Field.hpp"
#include "utils/StkHelpers.h"

namespace unit_test_alg_utils {

void
linear_scalar_field(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Field<double>& coordinates,
  stk::mesh::Field<double>& field,
  const double xCoeff,
  const double yCoeff,
  const double zCoeff)
{
  const stk::mesh::Selector sel = bulk.mesh_meta_data().universal_part();

  stk::mesh::EntityVector nodes;
  bulk.get_entities(stk::topology::NODE_RANK, sel, nodes);

  const unsigned fieldLength = sierra::nalu::max_extent(field, 0);
  if (fieldLength == 1) {
    for (stk::mesh::Entity& node : nodes) {
      double* fieldData = stk::mesh::field_data(field, node);
      double* coordsData = stk::mesh::field_data(coordinates, node);
      fieldData[0] = coordsData[0] * xCoeff + coordsData[1] * yCoeff +
                     coordsData[2] * zCoeff;
    }
  } else if (fieldLength == 3) {
    for (stk::mesh::Entity& node : nodes) {
      double* fieldData = stk::mesh::field_data(field, node);
      double* coordsData = stk::mesh::field_data(coordinates, node);
      fieldData[0] = coordsData[0] * xCoeff;
      fieldData[1] = coordsData[1] * yCoeff;
      fieldData[2] = coordsData[2] * zCoeff;
    }
  } else {
    STK_ThrowErrorMsg(
      "linear_scalar_field(): Field has unhandled length " << fieldLength);
  }

  field.modify_on_host();
}

} // namespace unit_test_alg_utils
