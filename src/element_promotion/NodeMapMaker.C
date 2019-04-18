/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/NodeMapMaker.h>
#include <element_promotion/ElementDescription.h>

#include <stk_util/util/ReportHandler.hpp>

#include <array>

namespace sierra {
namespace nalu{


Kokkos::View<int***> exodus_hex27_node_map()
{
  std::array<int,27> stkNodeMap = {{
      0,  8,  1, // bottom front edge
      11, 21,  9, // bottom mid-front edge
      3, 10,  2, // bottom back edge
      12, 25, 13, // mid-top front edge
      23, 20, 24, // mid-top mid-front edge
      15, 26, 14, // mid-top back edge
      4, 16,  5, // top front edge
      19, 22, 17, // top mid-front edge
      7, 18,  6  // top back edge
  }};

  Kokkos::View<int***> node_map("node_map", 3,3,3);
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        node_map(k,j,i) = stkNodeMap[9 * k + 3 * j + i];
      }
    }
  }
  return node_map;
}

Kokkos::View<int***> make_node_map_hex(int p, bool isPromoted)
{
  if (!isPromoted) return exodus_hex27_node_map();

  const int nodes1D = p + 1;
  Kokkos::View<int***> node_map("node_map", nodes1D, nodes1D, nodes1D);
  auto desc = ElementDescription::create(3, p);
  for (int k = 0; k < nodes1D; ++k) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int i = 0; i < nodes1D; ++i) {
        node_map(k,j,i) = desc->node_map(i,j,k);
      }
    }
  }
  return node_map;
}

Kokkos::View<int**> make_node_map_quad(int p)
{
  const int nodes1D = p + 1;
  Kokkos::View<int**> node_map("node_map", nodes1D, nodes1D);
  auto desc = ElementDescription::create(2, p);
    for (int j = 0; j < nodes1D; ++j) {
      for (int i = 0; i < nodes1D; ++i) {
        node_map(j,i) = desc->node_map(i,j);
      }
    }
  return node_map;
}

Kokkos::View<int***> make_face_node_map_hex(int p)
{
  const int nodes1D = p + 1;
  Kokkos::View<int***> face_node_map("face_node_map", 6, nodes1D, nodes1D);
  auto desc = ElementDescription::create(3, p);
  for (int faceOrdinal = 0; faceOrdinal < 6; ++faceOrdinal) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int i = 0; i < nodes1D; ++i) {
        face_node_map(faceOrdinal, j, i) = desc->faceNodeMap[faceOrdinal][i + j * nodes1D];
      }
    }
  }
  return face_node_map;
}

Kokkos::View<int**> make_face_node_map_quad(int p)
{
  const int nodes1D = p + 1;
  Kokkos::View<int**> face_node_map("face_node_map", 4, nodes1D);
  auto desc = ElementDescription::create(2, p);
  for (int faceOrdinal = 0; faceOrdinal < 4; ++faceOrdinal) {
    for (int i = 0; i < nodes1D; ++i) {
      face_node_map(faceOrdinal, i) = desc->faceNodeMap[faceOrdinal][i];
    }
  }
  return face_node_map;
}


Kokkos::View<int**> make_side_node_ordinal_map_hex(int p)
{
  const int nodes1D = p + 1;
  Kokkos::View<int**> face_node_map("side_node_ordinal_map", 6, nodes1D * nodes1D);
  auto desc = ElementDescription::create(3, p);
  for (int faceOrdinal = 0; faceOrdinal < 6; ++faceOrdinal) {
    for (int i = 0; i < nodes1D*nodes1D; ++i) {
      face_node_map(faceOrdinal, i) = desc->sideOrdinalMap[faceOrdinal][i];
  }}
  return face_node_map;
}

Kokkos::View<int**> make_side_node_ordinal_map_quad(int p)
{
  const int nodes1D = p + 1;
  Kokkos::View<int**> face_node_map("side_node_ordinal_map", 4, nodes1D);
  auto desc = ElementDescription::create(2, p);
  for (int faceOrdinal = 0; faceOrdinal < 4; ++faceOrdinal) {
    for (int i = 0; i < nodes1D; ++i) {
      face_node_map(faceOrdinal, i) = desc->sideOrdinalMap[faceOrdinal][i];
    }
  }
  return face_node_map;
}

Kokkos::View<int*> make_node_map_edge(int p)
{
  const int nodes1D = p +1;
  Kokkos::View<int*> edge_node_map("edge_node_map", nodes1D);
  edge_node_map(0) = 0;
  edge_node_map(p) = 1;
  for(int i = 1; i<p; i++){
    edge_node_map(i) = i+1;
  }
  return edge_node_map;
}


} // namespace nalu
} // namespace Sierra


