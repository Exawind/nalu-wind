// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <element_promotion/HexNElementDescription.h>

#include <stk_util/util/ReportHandler.hpp>
#include <stk_topology/topology.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <array>
#include <numeric>

namespace sierra {
namespace nalu {

//   Linear 8-Node Hexahedron node locations.
//
//          7                    6
//           o------------------o
//          /|                 /|
//         / |                / |
//        /  |               /  |
//       /   |              /   |
//      /    |             /    |
//     /     |            /     |
//  4 /      |         5 /      |
//   o------------------o       |
//   |       |          |       |
//   |     3 o----------|-------o 2
//   |      /           |      /
//   |     /            |     /
//   |    /             |    /
//   |   /              |   /
//   |  /               |  /
//   | /                | /
//   |/                 |/
//   o------------------o
//  0                    1
//
//----------------------------------------------
//           7         18         6
//            o--------o---------o
//           /|                 /|
//          / |                / |
//         /  |               /  |
//      19o   |            17o   |
//       /  15o             /    o14
//      /     |            /     |
//   4 /      | 16        /      |
//    o---------o--------o 5     |
//    |       |       10 |       |
//    |     3 o-------o--|-------o 2
//    |      /           |      /
//    |     /            |     /
//  12o    /             o13  /
//    |   o11            |   o9
//    |  /               |  /
//    | /                | /
//    |/                 |/
//    o---------o--------o
//   0          8         1
//
//----------------------------------------------
//
//            x--------x---------x
//           /|                 /|
//          / |                / |
//         /  |   21          /  |
//        x   |    o         x   |
//       /    x       o25   /    x     Node #26 is at centroid of element
//       (different from P=2 exodus)
//      /     |            /     |
//     /      |           /      |     "QUAD_9" beginning with nodes
//    x---------x--------x       |      0,1,5,4 (face_ordinal 0) has node 24 at
//    center.... | 22o   |          |   o23 | |       x-------x--|-------x | /
//    |      / |     /  24        |     / x    /    o        x    / |   x o20 |
//    x |  /               |  / | /                | /
//    |/                 |/
//    x---------x--------x
//
// And so on . . .

HexNElementDescription::HexNElementDescription(int p)
  : polyOrder(p),
    nodes1D(p + 1),
    nodesPerSide(nodes1D * nodes1D),
    nodesPerElement(nodes1D * nodes1D * nodes1D),
    newNodesPerEdge(polyOrder - 1),
    newNodesPerFace(newNodesPerEdge * newNodesPerEdge),
    newNodesPerVolume(newNodesPerEdge * newNodesPerEdge * newNodesPerEdge),
    subElementsPerElement((nodes1D - 1) * (nodes1D - 1) * (nodes1D - 1))
{
  set_edge_node_connectivities();
  set_face_node_connectivities();
  set_volume_node_connectivities();
  set_tensor_product_node_mappings();
  set_boundary_node_mappings();
  set_subelement_connectivites();
}
//--------------------------------------------------------------------------
std::vector<int>
HexNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  const int numNewNodes = newNodesPerEdge * numEdges;
  std::vector<int> edgeNodeOrdinals(numNewNodes);

  const int firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(
    edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_edge_node_connectivities()
{
  auto edgeNodeOrdinals = edge_node_ordinals();
  const int edgeMap[numEdges] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7};
  auto beginIterator = edgeNodeOrdinals.begin();
  for (int edgeOrdinal = 0; edgeOrdinal < numEdges; ++edgeOrdinal) {
    auto endIterator = beginIterator + newNodesPerEdge;
    edgeNodeConnectivities[edgeMap[edgeOrdinal]] =
      std::vector<int>{beginIterator, endIterator};
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<int>
HexNElementDescription::face_node_ordinals()
{
  // base nodes -> edge nodes -> face nodes for node ordering
  const int numNewFaceNodes = newNodesPerFace * numFaces;
  std::vector<int> faceNodeOrdinals(numNewFaceNodes);
  const int firstfaceNodeNumber =
    nodesInBaseElement + numEdges * newNodesPerEdge;
  std::iota(
    faceNodeOrdinals.begin(), faceNodeOrdinals.end(), firstfaceNodeNumber);
  return faceNodeOrdinals;
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_face_node_connectivities()
{
  const auto faceNodeOrdinals = face_node_ordinals();

  // there's a disconnect between the exodus node ordering and face ordering,
  // the first "new" face node is entered on face #5 (4 in C-numbering).
  const int faceMap[numFaces] = {4, 5, 3, 1, 0, 2};
  auto beginIterator = faceNodeOrdinals.begin();
  for (int faceOrdinal = 0; faceOrdinal < numFaces; ++faceOrdinal) {
    auto endIterator = beginIterator + newNodesPerFace;
    faceNodeConnectivities[faceMap[faceOrdinal]] =
      std::vector<int>{beginIterator, endIterator};
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<int>
HexNElementDescription::volume_node_ordinals()
{
  // base nodes -> edge nodes -> face nodes -> volume nodes for node ordering
  int numNewVolumeNodes = newNodesPerVolume;
  std::vector<int> volumeNodeOrdinals(numNewVolumeNodes);

  int firstVolumeNodeNumber = nodesInBaseElement + numEdges * newNodesPerEdge +
                              numFaces * newNodesPerFace;
  std::iota(
    volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(),
    firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities = volume_node_ordinals();
}
//--------------------------------------------------------------------------
std::pair<int, int>
HexNElementDescription::get_edge_offsets(int i, int j, int k, int edge_ordinal)
{
  // just hard-code each edge's directionality on a case-by-case basis.
  // there's a more general solution to this, but no need for it.

  // index of the "left"-most node along an edge
  int il = 0;
  int jl = 0;
  int kl = 0;

  // index of the "right"-most node along an edge
  int ir = nodes1D - 1;
  int jr = nodes1D - 1;
  int kr = nodes1D - 1;

  int ix = 0;
  int iy = 0;
  int iz = 0;
  int stk_index = 0;

  switch (edge_ordinal) {
  case 0: {
    ix = il + (i + 1);
    iy = jl;
    iz = kl;
    stk_index = i;
    break;
  }
  case 1: {
    ix = ir;
    iy = jl + (j + 1);
    iz = kl;
    stk_index = j;
    break;
  }
  case 2: {
    ix = ir - (i + 1);
    iy = jr;
    iz = kl;
    stk_index = i;
    break;
  }
  case 3: {
    ix = il;
    iy = jr - (j + 1);
    iz = kl;
    stk_index = j;
    break;
  }
  case 4: {
    ix = il + (i + 1);
    iy = jl;
    iz = kr;
    stk_index = i;
    break;
  }
  case 5: {
    ix = ir;
    iy = jl + (j + 1);
    iz = kr;
    stk_index = j;
    break;
  }
  case 6: {
    ix = ir - (i + 1);
    iy = jr;
    iz = kr;
    stk_index = i;
    break;
  }
  case 7: {
    ix = il;
    iy = jr - (j + 1);
    iz = kr;
    stk_index = j;
    break;
  }
  case 8: {
    ix = il;
    iy = jl;
    iz = kl + (k + 1);
    stk_index = k;
    break;
  }
  case 9: {
    ix = ir;
    iy = jl;
    iz = kl + (k + 1);
    stk_index = k;
    break;
  }
  case 10: {
    ix = ir;
    iy = jr;
    iz = kl + (k + 1);
    stk_index = k;
    break;
  }
  case 11: {
    ix = il;
    iy = jr;
    iz = kl + (k + 1);
    stk_index = k;
    break;
  }
  }

  int tensor_index = ix + nodes1D * (iy + nodes1D * iz);
  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
std::pair<int, int>
HexNElementDescription::get_face_offsets(int i, int j, int k, int face_ordinal)
{
  // just hard-code each face's orientation on a case-by-case basis.
  // there's a more general solution to this, but no need for it AFAIK.

  // index of the "left"-most node along an edge
  int il = 0;
  int jl = 0;
  int kl = 0;

  // index of the "right"-most node along an edge
  int ir = nodes1D - 1;
  int jr = nodes1D - 1;
  int kr = nodes1D - 1;

  int ix = 0;
  int iy = 0;
  int iz = 0;

  int face_i = 0;
  int face_j = 0;

  switch (face_ordinal) {
  case 0: {
    ix = il + (i + 1);
    iy = jl;
    iz = kl + (k + 1);

    face_i = i;
    face_j = k;
    break;
  }
  case 1: {
    ix = ir;
    iy = jl + (j + 1);
    iz = kl + (k + 1);

    face_i = j;
    face_j = k;
    break;
  }
  case 2: {
    ix = ir - (i + 1);
    iy = jr;
    iz = kl + (k + 1);

    face_i = i;
    face_j = k;
    break;
  }
  case 3: {
    ix = il;
    iy = jl + (j + 1);
    iz = kl + (k + 1);

    face_i = k;
    face_j = j;
    break;
  }
  case 4: {
    ix = il + (i + 1);
    iy = jl + (j + 1);
    iz = kl;

    face_i = j;
    face_j = i;
    break;
  }
  case 5: {
    ix = il + (i + 1);
    iy = jl + (j + 1);
    iz = kr;

    face_i = i;
    face_j = j;
    break;
  }
  }

  int tensor_index = ix + nodes1D * (iy + nodes1D * iz);
  int stk_index = face_i + newNodesPerEdge * face_j;

  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_base_node_maps()
{
  nodeMap.resize(nodesPerElement);
  node_map(0, 0, 0) = 0;
  node_map(polyOrder, 0, 0) = 1;
  node_map(polyOrder, polyOrder, 0) = 2;
  node_map(0, polyOrder, 0) = 3;
  node_map(0, 0, polyOrder) = 4;
  node_map(polyOrder, 0, polyOrder) = 5;
  node_map(polyOrder, polyOrder, polyOrder) = 6;
  node_map(0, polyOrder, polyOrder) = 7;
}
//--------------------------------------------------------------------------

namespace {

std::array<std::vector<int>, 4>
edge_node_connectivities(int p, int start_index)
{
  constexpr int numEdges = 4;
  const int newNodesPerEdge = p - 1;
  const int numNewNodes = newNodesPerEdge * numEdges;

  std::vector<int> edgeNodeOrdinals(numNewNodes);
  std::iota(edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), start_index);

  std::array<std::vector<int>, numEdges> edgeNodeConnectivities;
  auto beginIterator = edgeNodeOrdinals.begin();
  for (int edgeOrdinal = 0; edgeOrdinal < numEdges; ++edgeOrdinal) {
    auto endIterator = beginIterator + newNodesPerEdge;
    edgeNodeConnectivities[edgeOrdinal] =
      std::vector<int>{beginIterator, endIterator};
    beginIterator = endIterator;
  }
  return edgeNodeConnectivities;
}

std::vector<int>
volume_node_connectivities(int p, int start_index)
{
  // only one volume to associate
  const int numNewNodes = (p - 1) * (p - 1);
  std::vector<int> volumeNodeOrdinals(numNewNodes);
  std::iota(volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(), start_index);
  return volumeNodeOrdinals;
}

int
tensor_edge_index(int p, int i, int j, int edge_ordinal)
{
  // we need to rotate the indexing of the edges depending on their orientation
  // when we move from indexing nodes from the edge to indexing nodes from the
  // face
  const int nodes1D = p + 1;
  switch (edge_ordinal) {
  case 0:
    return i + 1; // bottom, left-to-right
  case 1:
    return p + nodes1D * (j + 1); // right-side, bottom-to-top.
  case 2:
    return p - (i + 1) + nodes1D * p; // top, right-to-left
  case 3:
    return nodes1D * (p - (j + 1)); // left-side, top-to-bottom
  default:
    return -1;
  }
}

int
stk_edge_index(int i, int j, int edge_ordinal)
{
  return (edge_ordinal == 0 || edge_ordinal == 2) ? i : j;
}

std::vector<int>
node_map_bc(int p)
{
  std::vector<int> nodeMap((p + 1) * (p + 1));
  // vertices
  nodeMap[0] = 0;
  nodeMap[p] = 1;
  nodeMap[p + p * (p + 1)] = 2;
  nodeMap[p * (p + 1)] = 3;

  constexpr int numVertices = 4;

  const auto edgeNodeConnectivities = edge_node_connectivities(p, numVertices);
  for (int edgeOrdinal = 0; edgeOrdinal < 4; ++edgeOrdinal) {
    const auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);
    for (int j = 0; j < p - 1; ++j) {
      for (int i = 0; i < p - 1; ++i) {
        const int tensorOffset = tensor_edge_index(p, i, j, edgeOrdinal);
        const int stkOffset = stk_edge_index(i, j, edgeOrdinal);
        nodeMap.at(tensorOffset) = newNodeOrdinals.at(stkOffset);
      }
    }
  }

  const auto newVolumeNodes = volume_node_connectivities(
    p, numVertices + edgeNodeConnectivities.size() * (p - 1));
  for (int j = 0; j < p - 1; ++j) {
    for (int i = 0; i < p - 1; ++i) {
      nodeMap[(i + 1) + (p + 1) * (j + 1)] = newVolumeNodes.at(i + j * (p - 1));
    }
  }
  return nodeMap;
}

} // namespace

void
HexNElementDescription::set_boundary_node_mappings()
{
  nodeMapBC = node_map_bc(polyOrder);
  inverseNodeMapBC.resize(nodes1D * nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      inverseNodeMapBC[nodeMapBC[i + nodes1D * j]] = {{i, j}};
    }
  }

  // index of the "left"-most node along an edge
  int il = 0;
  int jl = 0;
  int kl = 0;
  // index of the "right"-most node along an edge
  int ir = nodes1D - 1;
  int jr = nodes1D - 1;
  int kr = nodes1D - 1;
  // face node ordinals, reordered according to
  // the face permutation
  std::vector<std::vector<int>> reorderedFaceNodeMap(6);
  for (int j = 0; j < 6; ++j) {
    reorderedFaceNodeMap.at(j).resize(nodesPerSide);
  }

  // front -- counterclockwise
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(0).at(m + nodes1D * n) = node_map(m, jl, n);
    }
  }

  // right-- counterclockwise
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(1).at(m + nodes1D * n) = node_map(ir, m, n);
    }
  }

  // back -- flipped
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(2).at(m + nodes1D * n) =
        node_map(nodes1D - m - 1, jr, n);
    }
  }

  // left -- clockwise
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(3).at(m + nodes1D * n) = node_map(il, n, m);
    }
  }

  // bottom -- clockwise
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(4).at(m + nodes1D * n) = node_map(n, m, kl);
    }
  }

  // top -- counterclockwise
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      reorderedFaceNodeMap.at(5).at(m + nodes1D * n) = node_map(m, n, kr);
    }
  }

  for (int face_ordinal = 0; face_ordinal < 6; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodes1D * nodes1D; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) =
        reorderedFaceNodeMap.at(face_ordinal).at(ords[0] + nodes1D * ords[1]);
    }
  }
}

void
HexNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    for (int edgeOrdinal = 0; edgeOrdinal < numEdges; ++edgeOrdinal) {
      auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);

      for (int k = 0; k < newNodesPerEdge; ++k) {
        for (int j = 0; j < newNodesPerEdge; ++j) {
          for (int i = 0; i < newNodesPerEdge; ++i) {
            auto offsets = get_edge_offsets(i, j, k, edgeOrdinal);
            nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
          }
        }
      }
    }

    for (int faceOrdinal = 0; faceOrdinal < numFaces; ++faceOrdinal) {
      const auto& newNodeOrdinals = faceNodeConnectivities.at(faceOrdinal);
      for (int k = 0; k < newNodesPerEdge; ++k) {
        for (int j = 0; j < newNodesPerEdge; ++j) {
          for (int i = 0; i < newNodesPerEdge; ++i) {
            auto offsets = get_face_offsets(i, j, k, faceOrdinal);
            nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
          }
        }
      }
    }

    for (int k = 0; k < newNodesPerEdge; ++k) {
      for (int j = 0; j < newNodesPerEdge; ++j) {
        for (int i = 0; i < newNodesPerEdge; ++i) {
          node_map(i + 1, j + 1, k + 1) = volumeNodeConnectivities.at(
            i + newNodesPerEdge * (j + newNodesPerEdge * k));
        }
      }
    }
  }

  // inverse map
  inverseNodeMap.resize(nodes1D * nodes1D * nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int k = 0; k < nodes1D; ++k) {
        inverseNodeMap[node_map(i, j, k)] = {{i, j, k}};
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_subelement_connectivites()
{
  subElementConnectivity.resize(subElementsPerElement);
  for (int k = 0; k < nodes1D - 1; ++k) {
    for (int j = 0; j < nodes1D - 1; ++j) {
      for (int i = 0; i < nodes1D - 1; ++i) {
        subElementConnectivity[i + (nodes1D - 1) * (j + (nodes1D - 1) * k)] = {
          {node_map(i + 0, j + 0, k + 0), node_map(i + 1, j + 0, k + 0),
           node_map(i + 1, j + 0, k + 1), node_map(i + 0, j + 0, k + 1),
           node_map(i + 0, j + 1, k + 0), node_map(i + 1, j + 1, k + 0),
           node_map(i + 1, j + 1, k + 1), node_map(i + 0, j + 1, k + 1)}};
      }
    }
  }
}

} // namespace nalu
} // namespace sierra
