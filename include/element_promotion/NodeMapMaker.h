// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NodeMapMaker_h
#define NodeMapMaker_h

#include <KokkosInterface.h>

namespace stk {
struct topology;
}

namespace sierra {
namespace nalu {

Kokkos::View<int***> make_node_map_hex(int, bool = false);
Kokkos::View<int***> make_inverse_node_map_hex(int, bool = false);
Kokkos::View<int**> make_node_map_quad(int);
Kokkos::View<int**> make_side_node_ordinal_map_hex(int);

} // namespace nalu
} // namespace sierra

#endif
