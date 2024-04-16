// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "overset/overset_utils.h"
#include "Realm.h"

namespace sierra {
namespace nalu {
namespace overset_utils {

std::vector<OversetFieldData>
get_overset_field_data(Realm& realm, std::vector<std::string> fnames)
{
  std::vector<OversetFieldData> fields;
  const auto& meta = realm.meta_data();
  const int row = 1;

  for (const auto& ff : fnames) {
    auto* fld = meta.get_field(stk::topology::NODE_RANK, ff);
    STK_ThrowAssert(fld != nullptr);

    const int col = fld->max_size();
    fields.emplace_back(fld, row, col);
  }

  return fields;
}

} // namespace overset_utils
} // namespace nalu
} // namespace sierra
