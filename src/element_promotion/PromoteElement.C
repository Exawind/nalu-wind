// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/PromoteElementImpl.h>

#include <NaluEnv.h>
#include <BucketLoop.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/HashEntityAndEntityKey.hpp>
#include <stk_mesh/base/CreateEdges.hpp>
#include <stk_mesh/base/CreateFaces.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/parallel/CommSparse.hpp>
#include <stk_util/parallel/ParallelComm.hpp>
#include <stk_util/util/ReportHandler.hpp>

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <limits>

namespace sierra {
namespace nalu {
namespace promotion {

std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
create_tensor_product_hex_elements(
  std::vector<double> nodeLocs1D,
  stk::mesh::BulkData& bulk,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& partsToBePromoted)
{
  STK_ThrowRequire(check_parts_for_promotion(partsToBePromoted));
  return impl::promote_elements_hex(
    nodeLocs1D, bulk, coordField, partsToBePromoted);
}

stk::mesh::PartVector
create_promoted_boundary_elements(
  int p, stk::mesh::BulkData& bulk, const stk::mesh::PartVector& parts)
{
  return impl::create_boundary_elements(p, bulk, parts);
}

} // namespace promotion
} // namespace nalu
} // namespace sierra
