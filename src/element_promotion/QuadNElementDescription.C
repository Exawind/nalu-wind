/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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

QuadNElementDescription::QuadNElementDescription(std::vector<double> in_nodeLocs)
: ElementDescription()
{
  nodeLocs1D = in_nodeLocs;
  polyOrder = nodeLocs1D.size()-1;
  nodes1D = nodeLocs1D.size();
  nodesPerSide = nodes1D;
  nodesPerElement = nodes1D*nodes1D;

  baseTopo = stk::topology::QUAD_4_2D;
  dimension = 2;
  numEdges = 4;
  numFaces = 0;
  numBoundaries = numEdges;
  nodesInBaseElement = baseTopo.num_nodes();
  nodesPerSubElement = baseTopo.num_nodes();
  baseEdgeConnectivity = { {0,1}, {1,2}, {2,3}, {3,0} };

  //first 4 nodes are base nodes.  Rest have been added.
  baseNodeOrdinals = {0,1,2,3};
  promotedNodeOrdinals.resize(nodesPerElement-nodesInBaseElement);
  std::iota(promotedNodeOrdinals.begin(), promotedNodeOrdinals.end(), 4);

  newNodesPerEdge = polyOrder - 1;
  newNodesPerVolume = (polyOrder - 1)*(polyOrder - 1);

  set_edge_node_connectivities();
  set_volume_node_connectivities();
  set_tensor_product_node_mappings();
  set_boundary_node_mappings();
  set_side_node_ordinals();
  set_isoparametric_coordinates();
  set_subelement_connectivites();
}
//--------------------------------------------------------------------------
std::vector<int> QuadNElementDescription::edge_node_ordinals()
{
  // base nodes -> edge nodes for node ordering
  int numNewNodes = newNodesPerEdge * numEdges;
  std::vector<ordinal_type> edgeNodeOrdinals(numNewNodes);

  ordinal_type firstEdgeNodeNumber = nodesInBaseElement;
  std::iota(edgeNodeOrdinals.begin(), edgeNodeOrdinals.end(), firstEdgeNodeNumber);

  return edgeNodeOrdinals;
}
//--------------------------------------------------------------------------
void QuadNElementDescription::set_edge_node_connectivities()
{
  std::array<ordinal_type,4> edgeOrdinals = {{0, 1, 2, 3}};
  auto edgeNodeOrdinals = edge_node_ordinals();

  int edgeOffset = 0;
  for (const auto edgeOrdinal : edgeOrdinals) {
    std::vector<ordinal_type> newNodesOnEdge(polyOrder-1);
    for (int j = 0; j < polyOrder-1; ++j) {
      newNodesOnEdge.at(j) = edgeNodeOrdinals.at(edgeOffset + j);
    }
    edgeNodeConnectivities.insert({edgeOrdinal, newNodesOnEdge});
    edgeOffset += newNodesPerEdge;
  }
}
//--------------------------------------------------------------------------
std::vector<ordinal_type> QuadNElementDescription::volume_node_ordinals()
{
  // 2D volume
  int numNewNodes = (polyOrder-1) * (polyOrder-1);
  std::vector<ordinal_type> volumeNodeOrdinals(numNewNodes);

  ordinal_type firstVolumeNodeNumber = edgeNodeConnectivities.size() * (polyOrder-1) + nodesInBaseElement;
  std::iota(volumeNodeOrdinals.begin(), volumeNodeOrdinals.end(), firstVolumeNodeNumber);

  return volumeNodeOrdinals;
}
//--------------------------------------------------------------------------
void QuadNElementDescription::set_volume_node_connectivities()
{
  // Only 1 volume: just insert.
  volumeNodeConnectivities.insert({0, volume_node_ordinals()});
}
//--------------------------------------------------------------------------
std::pair<ordinal_type,ordinal_type>
QuadNElementDescription::get_edge_offsets(
  ordinal_type i, ordinal_type j,
  ordinal_type edge_ordinal)
{
  // index of the "left"-most node along an edge
  ordinal_type il = 0;
  ordinal_type jl = 0;

  // index of the "right"-most node along an edge
  ordinal_type ir = nodes1D - 1;
  ordinal_type jr = nodes1D - 1;

  // output
  ordinal_type ix = -1;
  ordinal_type iy = -1;
  ordinal_type stk_index = -1;

  // just hard-code
  switch (edge_ordinal) {
    case 0:
    {
      ix = il + (i + 1);
      iy = jl;
      stk_index = i;
      break;
    }
    case 1:
    {
      ix = ir;
      iy = jl + (j + 1);
      stk_index = j;
      break;
    }
    case 2:
    {
      ix = ir - (i + 1);
      iy = jr;
      stk_index = i;
      break;
    }
    case 3:
    {
      ix = il;
      iy = jr - (j + 1);
      stk_index = j;
      break;
    }
  }
  ordinal_type tensor_index = (ix + nodes1D * iy);;
  return {tensor_index, stk_index};
}
//--------------------------------------------------------------------------
void QuadNElementDescription::set_base_node_maps()
{
  nodeMap.resize(nodesPerElement);
  inverseNodeMap.resize(nodesPerElement);

  nmap(0        , 0        ) = 0;
  nmap(polyOrder, 0        ) = 1;
  nmap(polyOrder, polyOrder) = 2;
  nmap(0        , polyOrder) = 3;

  inmap(0) = {0        , 0        };
  inmap(1) = {polyOrder, 0        };
  inmap(2) = {polyOrder, polyOrder};
  inmap(3) = {0        , polyOrder};
}
void QuadNElementDescription::set_boundary_node_mappings()
{
  std::vector<ordinal_type> bcNodeOrdinals(polyOrder-1);
  std::iota(bcNodeOrdinals.begin(), bcNodeOrdinals.end(), 2);

  nodeMapBC.resize(nodes1D);
  nodeMapBC[0] = 0;
  for (int j = 1; j < polyOrder; ++j) {
    nodeMapBC.at(j) = bcNodeOrdinals.at(j-1);
  }
  nodeMapBC[nodes1D-1] = 1;

  inverseNodeMapBC.resize(nodes1D);
  for (int j = 0; j < nodes1D; ++j) {
    inverseNodeMapBC[node_map_bc(j)] = { j };
  }
}
//--------------------------------------------------------------------------
void QuadNElementDescription::set_tensor_product_node_mappings()
{
  set_base_node_maps();

  if (polyOrder > 1) {
    std::array<ordinal_type,4> edgeOrdinals = {{0, 1, 2, 3}};
    for (auto edgeOrdinal : edgeOrdinals) {
      auto newNodeOrdinals = edgeNodeConnectivities.at(edgeOrdinal);
      for (int j = 0; j < newNodesPerEdge; ++j) {
        for (int i = 0; i < newNodesPerEdge; ++i) {
          auto offsets = get_edge_offsets(i,j,edgeOrdinal);
          nodeMap.at(offsets.first) = newNodeOrdinals.at(offsets.second);
        }
      }
    }

    auto newVolumeNodes = volumeNodeConnectivities.at(0);
    for (int j = 0; j < polyOrder-1; ++j) {
      for (int i = 0; i < polyOrder-1; ++i) {
        nmap(i + 1, j + 1) = newVolumeNodes.at(i + j * (polyOrder - 1));
      }
    }
  }

  //inverse map
  inverseNodeMap.resize(nodes1D*nodes1D);
  for (int i = 0; i < nodes1D; ++i) {
    for (int j = 0; j < nodes1D; ++j) {
      inverseNodeMap[node_map(i,j)] = {i, j};
    }
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_isoparametric_coordinates()
{
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      std::vector<double> nodeLoc = { nodeLocs1D.at(i), nodeLocs1D.at(j) };
      nodeLocs.insert({node_map(i,j), nodeLoc});
    }
  }
}
//--------------------------------------------------------------------------
void
QuadNElementDescription::set_subelement_connectivites()
{
  subElementConnectivity.resize((nodes1D-1)*(nodes1D-1));
  for (int j = 0; j < nodes1D-1; ++j) {
    for (int i = 0; i < nodes1D-1; ++i) {
      subElementConnectivity[i+(nodes1D-1)*j]= {
          node_map(i+0,j+0),
          node_map(i+1,j+0),
          node_map(i+1,j+1),
          node_map(i+0,j+1)
      };
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
    faceNodeMap.at(2).at(m) = node_map(nodes1D-m-1, jr);
  }

  //left
  for (int m = 0; m < nodes1D; ++m) {
    faceNodeMap.at(3).at(m) = node_map(il, nodes1D-m-1);
  }

  sideOrdinalMap.resize(4);
  for (int face_ordinal = 0; face_ordinal < 4; ++face_ordinal) {
    sideOrdinalMap[face_ordinal].resize(nodesPerSide);
    for (int j = 0; j < nodesPerSide; ++j) {
      const auto& ords = inverseNodeMapBC[j];
      sideOrdinalMap.at(face_ordinal).at(j) = faceNodeMap.at(face_ordinal).at(ords[0]);
    }
  }
}

} // namespace nalu
}  // namespace sierra
