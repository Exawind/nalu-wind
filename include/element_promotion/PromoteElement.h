// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#ifndef PromoteElement_h
#define PromoteElement_h

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_topology/topology.hpp>
#include <FieldTypeDef.h>

#include <vector>

namespace stk {
namespace mesh {
class Part;
}
} // namespace stk
namespace stk {
namespace mesh {
typedef std::vector<Part*> PartVector;
}
} // namespace stk
namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace promotion {

std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
create_tensor_product_hex_elements(
  std::vector<double> nodeLocs1D,
  stk::mesh::BulkData& bulk,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& elemPartsToBePromoted);

stk::mesh::PartVector create_promoted_boundary_elements(
  int p, stk::mesh::BulkData& bulk, const stk::mesh::PartVector& meshParts);

} // namespace promotion
} // namespace nalu
} // namespace sierra

#endif
