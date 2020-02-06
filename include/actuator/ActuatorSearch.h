// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORSEARCH_H_
#define ACTUATORSEARCH_H_

#include <actuator/ActuatorTypes.h>
#include <actuator/ActuatorBulk.h>
#include <stk_mesh/base/BulkData.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Ghosting.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_search/IdentProc.hpp>

// common type defs
typedef stk::search::IdentProc<uint64_t, int> theKey;
typedef stk::search::Point<double> Point;
typedef stk::search::Sphere<double> Sphere;
typedef stk::search::Box<double> Box;
typedef std::pair<Sphere, theKey> boundingSphere;
typedef std::pair<Box, theKey> boundingElementBox;
using VecBoundSphere = std::vector<boundingSphere>;
using VecBoundElemBox = std::vector<boundingElementBox>;
using SearchKeyPair = std::pair<theKey, theKey>;
using VecSearchKeyPair = std::vector<SearchKeyPair>;

namespace sierra{
namespace nalu{

// This should be callable inside a functor?
// Make a kokkos function
VecBoundSphere CreateBoundingSpheres( ActFixVectorDbl points, ActFixScalarDbl searchRadius);

// Can this use NGP mesh with host only execution?
VecBoundElemBox CreateElementBoxes(stk::mesh::MetaData& stkMeta,
  stk::mesh::BulkData& stkBulk,
  std::vector<std::string> partNameList);

// potential overload if coarse search changes
// return element where id exists
VecSearchKeyPair ExecuteCoarseSearch(VecBoundSphere&, VecBoundElemBox&, stk::search::SearchMethod searchMethod);
ActFixScalarBool ExecuteFineSearch(stk::mesh::MetaData& stkMeta,
  stk::mesh::BulkData& stkBulk, VecSearchKeyPair& coarseResults, ActFixVectorDbl points, ActFixElemIds matchElemIds);

//TODO(psakiev) Wrapper function to call with ActuatorBulk



} //namespace nalu
} //namespace sierra

#endif // ACTUATORSEARCH_H_
