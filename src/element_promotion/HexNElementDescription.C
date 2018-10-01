/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/HexNElementDescription.h>
#include <element_promotion/QuadNElementDescription.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/QuadratureRule.h>
#include <element_promotion/TensorProductQuadratureRule.h>
#include <element_promotion/ElementDescription.h>
#include <NaluEnv.h>
#include <nalu_make_unique.h>

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
  //
  //
  //            x--------x---------x
  //           /|                 /|
  //          / |                / |
  //         /  |   21          /  |
  //        x   |    o         x   |
  //       /    x       o25   /    x     Node #26 is at centroid of element (different from P=2 exodus)
  //      /     |            /     |
  //     /      |           /      |     "QUAD_9" beginning with nodes
  //    x---------x--------x       |      0,1,5,4 (face_ordinal 0) has node 24 at center....
  //    | 22o   |          |   o23 |
  //    |       x-------x--|-------x
  //    |      /           |      /
  //    |     /  24        |     /
  //    x    /    o        x    /
  //    |   x        o20   |   x
  //    |  /               |  /
  //    | /                | /
  //    |/                 |/
  //    x---------x--------x
  //
  // And so on . . .

HexNElementDescription::HexNElementDescription(std::vector<double> in_nodeLocs)
: ElementDescription()
{
  nodeLocs1D = in_nodeLocs;

  baseTopo = stk::topology::HEX_8;
  polyOrder = nodeLocs1D.size()-1;
  nodes1D = nodeLocs1D.size();
  nodesPerSide = nodes1D * nodes1D;
  nodesPerElement = nodes1D * nodes1D * nodes1D;
  dimension = baseTopo.dimension();
  numEdges = baseTopo.num_edges();
  numFaces = baseTopo.num_faces();
  numBoundaries = numFaces;
  nodesInBaseElement = baseTopo.num_nodes();
  nodesPerSubElement = nodesInBaseElement;

  baseEdgeConnectivity = {
      {0,1}, {1,2}, {2,3}, {3,0}, // bottom face
      {4,5}, {5,6}, {6,7}, {7,4}, // top face
      {0,4}, {1,5}, {2,6}, {3,7}  // bottom-to-top
  };

  baseFaceConnectivity = {
      {0, 1, 5, 4}, // front face
      {1, 2, 6, 5}, // right face
      {2, 3, 7, 6}, // back face
      {0, 4, 7, 3}, // left face
      {0, 3, 2, 1}, // bottom face
      {4, 5, 6, 7}  // top face
  };

  baseFaceEdgeConnectivity = {
      { 0,  9,  4,  8},
      { 1, 10,  5,  9},
      { 2, 11,  6, 10},
      { 8,  7, 11,  3},
      { 3,  2,  1,  0},
      { 4,  5,  6,  7}
  };

  //first 8 nodes are base nodes.  Rest have been added.
  baseNodeOrdinals = {0,1,2,3,4,5,6,7};

  promotedNodeOrdinals.resize(nodesPerElement-nodesInBaseElement);
  std::iota(promotedNodeOrdinals.begin(), promotedNodeOrdinals.end(), nodesInBaseElement);

  newNodesPerEdge   = polyOrder - 1;
  newNodesPerFace   = newNodesPerEdge * newNodesPerEdge;
  newNodesPerVolume = newNodesPerEdge * newNodesPerEdge * newNodesPerEdge;

  set_edge_node_connectivities();
  set_face_node_connectivities();
  set_volume_node_connectivities();
  set_tensor_product_node_mappings();
  set_boundary_node_mappings();
  set_side_node_ordinals();
  set_isoparametric_coordinates();
  set_subelement_connectivites();
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> HexNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  ordinal_type numNewNodes = newNodesPerEdge * numEdges;
  std::vector<ordinal_type> edgeNodeOrdinals(numNewNodes);

  ordinal_type firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_edge_node_connectivities()
{
  std::vector<ordinal_type> edgeOrdinals(numEdges);
  std::iota(edgeOrdinals.begin(), edgeOrdinals.end(), 0);

  auto edgeNodeOrdinals = edge_node_ordinals();
  ordinal_type edgeMap[12] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7 };

  auto beginIterator = edgeNodeOrdinals.begin();
  for (const auto edgeOrdinal : edgeOrdinals) {
    auto endIterator = beginIterator + newNodesPerEdge;
    edgeNodeConnectivities.insert({edgeMap[edgeOrdinal], std::vector<ordinal_type>{beginIterator,endIterator}});
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> HexNElementDescription::face_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  ordinal_type numNewFaceNodes = newNodesPerFace * numFaces;
  std::vector<ordinal_type> faceNodeOrdinals(numNewFaceNodes);

  ordinal_type firstfaceNodeNumber = nodesInBaseElement + numEdges * newNodesPerEdge;
  std::iota(faceNodeOrdinals.begin(), faceNodeOrdinals.end(), firstfaceNodeNumber);

  return faceNodeOrdinals;
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_face_node_connectivities()
{
  std::vector<ordinal_type> faceOrdinals(numFaces);
  std::iota(faceOrdinals.begin(), faceOrdinals.end(), 0);

  auto faceNodeOrdinals = face_node_ordinals();

  // there's a disconnect between the exodus node ordering and face ordering,
  // the first "new" face node is entered on face #5 (4 in C-numbering).
  ordinal_type faceMap[6] = { 4, 5, 3, 1, 0, 2 };

  auto beginIterator = faceNodeOrdinals.begin();
  for (const auto faceOrdinal : faceOrdinals) {
    auto endIterator = beginIterator + newNodesPerFace;
    faceNodeConnectivities.insert({faceMap[faceOrdinal], std::vector<ordinal_type>{beginIterator,endIterator}});
    beginIterator = endIterator;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> HexNElementDescription::volume_node_ordinals()
{
  // 3D volume
  ordinal_type numNewVolumeNodes = newNodesPerVolume;
  std::vector<ordinal_type> volumeNodeOrdinals(numNewVolumeNodes);

  ordinal_type firstVolumeNodeNumber = nodesInBaseElement + numEdges * newNodesPerEdge + numFaces * newNodesPerFace;
  std::iota(volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(), firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities.insert({0, volume_node_ordinals()});
}
//--------------------------------------------------------------------------
std::pair<ordinal_type,ordinal_type>
HexNElementDescription::get_edge_offsets(
  ordinal_type i, ordinal_type j, ordinal_type k,
  ordinal_type edge_ordinal)
{
  // just hard-code each edge's directionality on a case-by-case basis.
  // there's a more general solution to this, but no need for it.

  // index of the "left"-most node along an edge
  ordinal_type il = 0;
  ordinal_type jl = 0;
  ordinal_type kl = 0;

  // index of the "right"-most node along an edge
  ordinal_type ir = nodes1D - 1;
  ordinal_type jr = nodes1D - 1;
  ordinal_type kr = nodes1D - 1;

  ordinal_type ix = 0;
  ordinal_type iy = 0;
  ordinal_type iz = 0;
  ordinal_type stk_index = 0;

  switch (edge_ordinal)
  {
    case 0:
    {
      ix = il+(i+1);
      iy = jl;
      iz = kl;
      stk_index = i;
      break;
    }
    case 1:
    {
      ix = ir;
      iy = jl + (j + 1);
      iz = kl;
      stk_index = j;
      break;
    }
    case 2:
    {
      ix = ir - (i + 1);
      iy = jr;
      iz = kl;
      stk_index = i;
      break;
    }
    case 3:
    {
      ix = il;
      iy = jr - (j + 1);
      iz = kl;
      stk_index = j;
      break;
    }
    case 4:
    {
      ix = il+(i+1);
      iy = jl;
      iz = kr;
      stk_index = i;
      break;
    }
    case 5:
    {
      ix = ir;
      iy = jl + (j + 1);
      iz = kr;
      stk_index = j;
      break;
    }
    case 6:
    {
      ix = ir - (i + 1);
      iy = jr;
      iz = kr;
      stk_index = i;
      break;
    }
    case 7:
    {
      ix = il;
      iy = jr - (j + 1);
      iz = kr;
      stk_index = j;
      break;
    }
    case 8:
    {
      ix = il;
      iy = jl;
      iz = kl + (k + 1);
      stk_index = k;
      break;
    }
    case 9:
    {
      ix = ir;
      iy = jl;
      iz = kl + (k + 1);
      stk_index = k;
      break;
    }
    case 10:
    {
      ix = ir;
      iy = jr;
      iz = kl + (k + 1);
      stk_index = k;
      break;
    }
    case 11:
    {
      ix = il;
      iy = jr;
      iz = kl + (k + 1);
      stk_index = k;
      break;
    }
  }

  ordinal_type tensor_index = ix + nodes1D * (iy + nodes1D * iz);
  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
std::pair<ordinal_type,ordinal_type>
HexNElementDescription::get_face_offsets(
  ordinal_type i, ordinal_type j, ordinal_type k,
  ordinal_type face_ordinal)
{
  // just hard-code each face's orientation on a case-by-case basis.
  // there's a more general solution to this, but no need for it AFAIK.

  // index of the "left"-most node along an edge
  ordinal_type il = 0;
  ordinal_type jl = 0;
  ordinal_type kl = 0;

  // index of the "right"-most node along an edge
  ordinal_type ir = nodes1D - 1;
  ordinal_type jr = nodes1D - 1;
  ordinal_type kr = nodes1D - 1;

  ordinal_type ix = 0;
  ordinal_type iy = 0;
  ordinal_type iz = 0;

  ordinal_type face_i = 0;
  ordinal_type face_j = 0;

  switch (face_ordinal) {
    case 0:
    {
      ix = il + (i + 1);
      iy = jl;
      iz = kl + (k + 1);

      face_i = i;
      face_j = k;
      break;
    }
    case 1:
    {
      ix = ir;
      iy = jl + (j + 1);
      iz = kl + (k + 1);

      face_i = j;
      face_j = k;
      break;
    }
    case 2:
    {
      ix = ir - (i + 1);
      iy = jr;
      iz = kl + (k + 1);

      face_i = i;
      face_j = k;
      break;
    }
    case 3:
    {
      ix = il;
      iy = jl + (j+1);
      iz = kl + (k+1);

      face_i = k;
      face_j = j;
      break;
    }
    case 4:
    {
      ix = il + (i+1);
      iy = jl + (j+1);
      iz = kl;

      face_i = j;
      face_j = i;
      break;
    }
    case 5:
    {
      ix = il + (i+1);
      iy = jl + (j+1);
      iz = kr;

      face_i = i;
      face_j = j;
      break;
    }
  }

  ordinal_type tensor_index = ix + nodes1D * (iy + nodes1D * iz);
  ordinal_type stk_index = face_i + newNodesPerEdge * face_j;

  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_base_node_maps()
{
  nodeMap.resize(nodesPerElement);
  inverseNodeMap.resize(nodesPerElement);

  nmap(0        , 0        , 0        ) = 0;
  nmap(polyOrder, 0        , 0        ) = 1;
  nmap(polyOrder, polyOrder, 0        ) = 2;
  nmap(0        , polyOrder, 0        ) = 3;
  nmap(0        , 0        , polyOrder) = 4;
  nmap(polyOrder, 0        , polyOrder) = 5;
  nmap(polyOrder, polyOrder, polyOrder) = 6;
  nmap(0        , polyOrder, polyOrder) = 7;
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_boundary_node_mappings()
{
  // node mapping needs to be consistent with quad element's
  nodeMapBC = QuadNElementDescription(nodeLocs1D).nodeMap;

  inverseNodeMap.resize(nodes1D*nodes1D*nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int k = 0; k < nodes1D; ++k) {
        inverseNodeMap[node_map(i,j,k)] = {i, j, k};
      }
    }
  }

  inverseNodeMapBC.resize(nodes1D*nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      inverseNodeMapBC[node_map_bc(i,j)] = { i,j };
    }
  }
}
//--------------------------------------------------------------------------
void HexNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    for (int edgeOrdinal = 0; edgeOrdinal < numEdges; ++edgeOrdinal) {
      auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);

      for (int k = 0; k < newNodesPerEdge; ++k) {
        for (int j = 0; j < newNodesPerEdge; ++j) {
          for (int i = 0; i < newNodesPerEdge; ++i) {
            auto offsets = get_edge_offsets(i,j,k,edgeOrdinal);
            nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
          }
        }
      }
    }

    for (int faceOrdinal = 0; faceOrdinal < numFaces; ++faceOrdinal) {
      auto newNodeOrdinals = faceNodeConnectivities.at(faceOrdinal);
      for (int k = 0; k < newNodesPerEdge; ++k) {
        for (int j = 0; j < newNodesPerEdge; ++j) {
          for (int i = 0; i < newNodesPerEdge; ++i) {
            auto offsets = get_face_offsets(i,j,k,faceOrdinal);
            nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
          }
        }
      }
    }

    auto newVolumeNodes = volumeNodeConnectivities.at(0);
    for (int k = 0; k < newNodesPerEdge; ++k) {
      for (int j = 0; j < newNodesPerEdge; ++j) {
        for (int i = 0; i < newNodesPerEdge; ++i) {
          nmap(i + 1, j + 1, k + 1) = newVolumeNodes.at(i + newNodesPerEdge * (j + newNodesPerEdge * k));
        }
      }
    }
  }

  //inverse map
  inverseNodeMap.resize(nodes1D*nodes1D*nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int k = 0; k < nodes1D; ++k) {
        inverseNodeMap[node_map(i,j,k)] = {i, j, k};
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_isoparametric_coordinates()
{
  for (int k = 0; k < nodes1D; ++k) {
    for (int j = 0; j < nodes1D; ++j) {
      for (int i = 0; i < nodes1D; ++i) {
        std::vector<double> nodeLoc = { nodeLocs1D.at(i), nodeLocs1D.at(j), nodeLocs1D.at(k) };
        nodeLocs.insert({node_map(i,j,k), nodeLoc});
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_subelement_connectivites()
{
  subElementConnectivity.resize((nodes1D-1)*(nodes1D-1)*(nodes1D-1));
  for (int k = 0; k < nodes1D-1; ++k) {
    for (int j = 0; j < nodes1D-1; ++j) {
      for (int i = 0; i < nodes1D-1; ++i) {
        subElementConnectivity[i+(nodes1D-1)*(j+(nodes1D-1)*k)] =
        {
            node_map(i+0,j+0,k+0),
            node_map(i+1,j+0,k+0),
            node_map(i+1,j+0,k+1),
            node_map(i+0,j+0,k+1),
            node_map(i+0,j+1,k+0),
            node_map(i+1,j+1,k+0),
            node_map(i+1,j+1,k+1),
            node_map(i+0,j+1,k+1)
        };
      }
    }
  }
}
//--------------------------------------------------------------------------
void
HexNElementDescription::set_side_node_ordinals()
{
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
  std::vector<std::vector<int>> reorderedFaceNodeMap;

  faceNodeMap.resize(numBoundaries);
  reorderedFaceNodeMap.resize(numBoundaries);
  for (int j = 0; j < numBoundaries; ++j) {
    faceNodeMap.at(j).resize(nodesPerSide);
    reorderedFaceNodeMap.at(j).resize(nodesPerSide);
  }

  //front
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(0).at(m+nodes1D*n) = node_map(m,jl,n);
      reorderedFaceNodeMap.at(0).at(m+nodes1D*n) = node_map(m,jl,n);
    }
  }

  // right
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(1).at(m+nodes1D*n) = node_map(ir,m,n);
      reorderedFaceNodeMap.at(1).at(m+nodes1D*n) = node_map(ir,m,n);
    }
  }

  //back
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(2).at(m+nodes1D*n) = node_map(m,jr,n);
      reorderedFaceNodeMap.at(2).at(m+nodes1D*n) = node_map(nodes1D-m-1,jr,n);
    }
  }

  //left
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(3).at(m+nodes1D*n) = node_map(il,m,n);
      reorderedFaceNodeMap.at(3).at(m+nodes1D*n) = node_map(il,n,m);
    }
  }

  //bottom
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(4).at(m+nodes1D*n) = node_map(m,n,kl);
      reorderedFaceNodeMap.at(4).at(m+nodes1D*n) = node_map(n,m,kl);
    }
  }

  //top
  for (int n = 0; n < nodes1D; ++n) {
    for (int m = 0; m < nodes1D; ++m) {
      faceNodeMap.at(5).at(m+nodes1D*n) = node_map(m,n,kr);
      reorderedFaceNodeMap.at(5).at(m+nodes1D*n) = node_map(m,n,kr);
    }
  }

  sideOrdinalMap.resize(6);
  for (int face_ordinal = 0; face_ordinal < 6; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodes1D*nodes1D; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) = reorderedFaceNodeMap.at(face_ordinal).at(ords[0]+nodes1D*ords[1]);
    }
  }
}

} // namespace nalu
}  // namespace sierra
