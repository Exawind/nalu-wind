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

namespace sierra
{
namespace nalu
{

Kokkos::View<int***> make_node_map_hex(int, bool  = false);
Kokkos::View<int***> make_inverse_node_map_hex(int, bool = false);

Kokkos::View<int**> make_node_map_quad(int, bool  = false);

Kokkos::View<int***> make_face_node_map_hex(int);
Kokkos::View<int**> make_face_node_map_quad(int);

Kokkos::View<int**> make_side_node_ordinal_map_hex(int);
Kokkos::View<int**> make_side_node_ordinal_map_quad(int);



} // namespace nalu
} // namespace sierra

#endif
