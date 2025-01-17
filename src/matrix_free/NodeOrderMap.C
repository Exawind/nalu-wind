// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/NodeOrderMap.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
constexpr StkNodeOrderMapping<1>::node_map_type StkNodeOrderMapping<1>::map;
constexpr StkNodeOrderMapping<2>::node_map_type StkNodeOrderMapping<2>::map;
constexpr StkNodeOrderMapping<3>::node_map_type StkNodeOrderMapping<3>::map;
constexpr StkNodeOrderMapping<4>::node_map_type StkNodeOrderMapping<4>::map;
constexpr StkFaceNodeMapping<1>::node_map_type StkFaceNodeMapping<1>::map;
constexpr StkFaceNodeMapping<2>::node_map_type StkFaceNodeMapping<2>::map;
constexpr StkFaceNodeMapping<3>::node_map_type StkFaceNodeMapping<3>::map;
constexpr StkFaceNodeMapping<4>::node_map_type StkFaceNodeMapping<4>::map;

int
node_map(int p, int n, int m, int l)
{
  switch (p) {
  case 2:
    return StkNodeOrderMapping<2>::map(n, m, l);
  case 3:
    return StkNodeOrderMapping<3>::map(n, m, l);
  case 4:
    return StkNodeOrderMapping<4>::map(n, m, l);
  default:
    return StkNodeOrderMapping<1>::map(n, m, l);
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
