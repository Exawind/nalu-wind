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
#include <stk_mesh/base/BulkData.hpp>
#include <Kokkos_Core.hpp>
#include <stk_search/BoundingBox.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/SearchMethod.hpp>

// common type defs
using theKey = stk::search::IdentProc<uint64_t, int>;
using Point = stk::search::Point<double>;
using Sphere = stk::search::Sphere<double>;
using Box = stk::search::Box<double>;
using boundingSphere = std::pair<Sphere, theKey>;
using boundingElementBox = std::pair<Box, theKey>;
using VecBoundSphere = std::vector<boundingSphere>;
using VecBoundElemBox = std::vector<boundingElementBox>;
using SearchKeyPair = std::pair<theKey, theKey>;
using VecSearchKeyPair = std::vector<SearchKeyPair>;

namespace sierra {
namespace nalu {

// should we pass these return variables by reference and make void functions?
// thoughts?
VecBoundSphere
CreateBoundingSpheres(ActFixVectorDbl points, ActFixScalarDbl searchRadius);

VecBoundElemBox CreateElementBoxes(
  stk::mesh::BulkData& stkBulk, std::vector<std::string> partNameList);

// potential overload if coarse search backend changes
VecSearchKeyPair ExecuteCoarseSearch(
  VecBoundSphere& spheres,
  VecBoundElemBox& elemBoxes,
  stk::search::SearchMethod searchMethod);

ActFixScalarBool ExecuteFineSearch(
  stk::mesh::BulkData& stkBulk,
  VecSearchKeyPair& coarseResults,
  ActFixVectorDbl points,
  ActFixElemIds matchElemIds,
  ActFixVectorDbl localCoords);

} // namespace nalu
} // namespace sierra

#endif // ACTUATORSEARCH_H_
