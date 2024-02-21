// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/AMSAveragesAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "EigenDecomposition.h"
#include "utils/AMSUtils.h"

namespace sierra {
namespace nalu {

AMSAveragesAlg::AMSAveragesAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part)
{
}
} // namespace nalu
} // namespace sierra
