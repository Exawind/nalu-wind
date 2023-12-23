// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOCAL_VOLUME_SEARCH_H
#define LOCAL_VOLUME_SEARCH_H

#include "stk_mesh/base/Field.hpp"

#include "stk_search/BoundingBox.hpp"
#include "stk_search/IdentProc.hpp"

#include <array>
#include <vector>

namespace stk {
namespace mesh {
class BulkData;
class Selector;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

// reusable allocation assuming fixed mesh graph/fixed number of points
struct LocalVolumeSearchData
{
  LocalVolumeSearchData(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& sel,
    int npoints);

  using sphere_t = stk::search::Sphere<double>;
  using box_t = stk::search::Box<double>;
  using ident_t = stk::search::IdentProc<stk::mesh::EntityId, int>;

  std::vector<std::pair<sphere_t, ident_t>> search_points;
  std::vector<std::pair<box_t, ident_t>> search_boxes;
  std::vector<std::pair<ident_t, ident_t>> search_matches;
  std::vector<std::array<double, 3>> interpolated_values;
  std::vector<double> dist;
  std::vector<int> ownership;
};

// interpolate to a collection of points locally
// if the process contains the point then that the element of the second return
// vector is marked 1
void local_field_interpolation(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const std::vector<std::array<double, 3>>& points,
  const stk::mesh::Field<double>& coord_field,
  const stk::mesh::Field<double>& field_nm1,
  const stk::mesh::Field<double>& field,
  double dtratio,
  LocalVolumeSearchData& data);

} // namespace nalu
} // namespace sierra

#endif
