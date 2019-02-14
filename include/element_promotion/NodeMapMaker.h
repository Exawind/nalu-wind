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

Kokkos::View<int*> make_node_map(int p, stk::topology topo);
Kokkos::View<int*> make_inverse_node_map(int p, stk::topology topo);

} // namespace nalu
} // namespace sierra

#endif
