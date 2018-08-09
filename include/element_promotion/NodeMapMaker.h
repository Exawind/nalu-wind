/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef NodeMapMaker_h
#define NodeMapMaker_h

#include <KokkosInterface.h>

namespace stk { struct topology; }

namespace sierra {
namespace nalu{

Kokkos::View<int*> make_node_map(int p, stk::topology base_topo, bool isPromoted = false);
Kokkos::View<int*> make_inverse_node_map(int p, stk::topology base_topo, bool isPromoted = false);
Kokkos::View<int*> make_node_map(int p, int dim, bool isPromoted = false);
Kokkos::View<int*> make_inverse_node_map(int p, int dim, bool isPromoted = false);

} // namespace nalu
} // namespace sierra

#endif
