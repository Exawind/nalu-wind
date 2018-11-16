/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "WallDistSrcNodeSuppAlg.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

WallDistSrcNodeSuppAlg::WallDistSrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    dualNodalVolume_(realm.meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "dual_nodal_volume"))
{}

void
WallDistSrcNodeSuppAlg::node_execute(
  double* , double* rhs, stk::mesh::Entity node)
{
  const double dualVol = *stk::mesh::field_data(*dualNodalVolume_, node);

  rhs[0] += dualVol;
}

}  // nalu
}  // sierra
