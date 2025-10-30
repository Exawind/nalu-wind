// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef OVERSET_UTILS_H
#define OVERSET_UTILS_H

#include "overset/OversetFieldData.h"
#include <vector>
#include <string>

namespace sierra {
namespace nalu {

class Realm;

namespace overset_utils {

std::vector<OversetFieldData>
get_overset_field_data(Realm&, std::vector<std::string> fnames);
}
} // namespace nalu
} // namespace sierra

#endif /* OVERSET_UTILS_H */
