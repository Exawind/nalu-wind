// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <element_promotion/QuadNElementDescription.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <array>
#include <numeric>

namespace sierra {
namespace nalu {

int
tensor_edge_index(int p, int i, int j, int edge_ordinal)
{
  // we need to rotate the indexing of the edges depending on their orientation
  // when we move from indexing nodes from the edge to indexing nodes from the
  // face
  switch (edge_ordinal) {
  case 0:
    return i + 1;
  case 1:
    return p + (p + 1) * (j + 1);
  case 2:
    return p - (i + 1) + (p + 1) * p;
  case 3:
    return (p + 1) * (p - (j + 1)) default : return -1;
  }
}

int
stk_edge_index(int p, int i, int j, int edge_ordinal)
{
  // is the edge along the "x" or "y" reference coordinate
  return (edge_ordinal == 0 || edge_ordinal == 2) ? i : j;
}

//--------------------------------------------------------------------------
std::vector<int>
QuadNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  int numNewNodes = newNodesPerEdge * numEdges;
  std::vector<int> edgeNodeOrdinals(numNewNodes);

  int firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(
    edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_edge_node_connectivities()
{
  std::array<int, 4> edgeOrdinals = {{0, 1, 2, 3}};
  auto edgeNodeOrdinals = edge_node_ordinals();

  int edgeOffset = 0;
  for (const auto edgeOrdinal : edgeOrdinals) {
    std::vector<int> newNodesOnEdge(polyOrder - 1);
    for (int j = 0; j < polyOrder - 1; ++j) {
      newNodesOnEdge.at(j) = edgeNodeOrdinals.at(edgeOffset + j);
    }
    edgeNodeConnectivities.insert({edgeOrdinal, newNodesOnEdge});
    edgeOffset += newNodesPerEdge;
  }
}
//--------------------------------------------------------------------------
std::vector<int>
QuadNElementDescription::volume_node_ordinals()
{
  // 2D volume
  int numNewNodes = (polyOrder - 1) * (polyOrder - 1);
  std::vector<int> volumeNodeOrdinals(numNewNodes);

  int firstVolumeNodeNumber =
    edgeNodeConnectivities.size() * (polyOrder - 1) + nodesInBaseElement;
  std::iota(
    volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(),
    firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities.insert({0, volume_node_ordinals()});
}
//--------------------------------------------------------------------------
std::pair<int, int>
QuadNElementDescription::get_edge_offsets(int i, int j, int edge_ordinal)
{
  // index of the "left"-most node along an edge
  int il = 0;
  int jl = 0;

  // index of the "right"-most node along an edge
  int ir = nodes1D - 1;
  int jr = nodes1D - 1;

  // output
  int ix = -1;
  int iy = -1;
  int stk_index = -1;

  // just hard-code
  switch (edge_ordinal) {
  case 0: {
    ix = il + (i + 1);
    iy = jl;
    stk_index = i;
    break;
  }
  case 1: {
    ix = ir;
    iy = jl + (j + 1);
    stk_index = j;
    break;
  }
  case 2: {
    ix = ir - (i + 1);
    iy = jr;
    stk_index = i;
    break;
  }
  case 3: {
    ix = il;
    iy = jr - (j + 1);
    stk_index = j;
    break;
  }
  }
  int tensor_index = (ix + nodes1D * iy);
  ;
  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_base_node_maps()
{
  nodeMap.resize(nodesPerElement);
  inverseNodeMap.resize(nodesPerElement);

  auto& nmap = [&nodeMap](int i, int j) { return i + nodes1D * j; };
  nmap(0, 0) = 0;
  nmap(polyOrder, 0) = 1;
  nmap(polyOrder, polyOrder) = 2;
  nmap(0, polyOrder) = 3;

  inmap(0) = {0, 0};
  inmap(1) = {polyOrder, 0};
  inmap(2) = {polyOrder, polyOrder};
  inmap(3) = {0, polyOrder};
}
void
QuadNElementDescription::set_boundary_node_mappings()
{
  std::vector<int> bcNodeOrdinals(polyOrder - 1);
  std::iota(bcNodeOrdinals.begin(), bcNodeOrdinals.end(), 2);

  nodeMapBC.resize(nodes1D);
  nodeMapBC[0] = 0;
  for (int j = 1; j < polyOrder; ++j) {
    nodeMapBC.at(j) = bcNodeOrdinals.at(j - 1);
  }
  nodeMapBC[nodes1D - 1] = 1;

  inverseNodeMapBC.resize(nodes1D);
  for (int j = 0; j < nodes1D; ++j) {
    inverseNodeMapBC[node_map_bc(j)] = {j};
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    std::array<int, 4> edgeOrdinals = {{0, 1, 2, 3}};
    for (auto edgeOrdinal : edgeOrdinals) {
      auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);
      for (int j = 0; j < newNodesPerEdge; ++j) {
        for (int i = 0; i < newNodesPerEdge; ++i) {
          auto offsets = get_edge_offsets(i, j, edgeOrdinal);
          nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
        }
      }
    }

    auto newVolumeNodes = volumeNodeConnectivities.at(0);
    for (int j = 0; j < polyOrder - 1; ++j) {
      for (int i = 0; i < polyOrder - 1; ++i) {
        nmap(i + 1, j + 1) = newVolumeNodes.at(i + j * (polyOrder - 1));
      }
    }
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_isoparametric_coordinates()
{
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      std::vector<double> nodeLoc = {nodeLocs1D.at(i), nodeLocs1D.at(j)};
      nodeLocs.insert({node_map(i, j), nodeLoc});
    }
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_subelement_connectivites()
{
  subElementConnectivity.resize((nodes1D - 1) * (nodes1D - 1));
  for (int j = 0; j < nodes1D - 1; ++j) {
    for (int i = 0; i < nodes1D - 1; ++i) {
      subElementConnectivity[i + (nodes1D - 1) * j] = {
        node_map(i + 0, j + 0), node_map(i + 1, j + 0), node_map(i + 1, j + 1),
        node_map(i + 0, j + 1)};
    }
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_side_node_ordinals()
{
  // index of the "left"-most node along an edge
  int il = 0;
  int jl = 0;

  // index of the "right"-most node along an edge
  int ir = nodes1D - 1;
  int jr = nodes1D - 1;

  // face node ordinals, reordered according to
  // the face permutation

  faceNodeMap.resize(numBoundaries);
  for (int face_ord = 0; face_ord < numBoundaries; ++face_ord) {
    faceNodeMap.at(face_ord).resize(nodesPerSide);
  }

  // bottom
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(0).at(m) = node_map(m, jl);
  }

  // right
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(1).at(m) = node_map(ir, m);
  }

  // top
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(2).at(m) = node_map(nodes1D - m - 1, jr);
  }

  // left
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(3).at(m) = node_map(il, nodes1D - m - 1);
  }

  sideOrdinalMap.resize(4);
  for (int face_ordinal = 0; face_ordinal < 4; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodesPerSide; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) =
        faceNodeMap.at(face_ordinal).at(ords[0]);
    }
  }
}

} // namespace nalu
} // namespace sierra
