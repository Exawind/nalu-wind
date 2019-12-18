// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef HexNElementDescription_h
#define HexNElementDescription_h

#include <stddef.h>
#include <map>
#include <memory>
#include <vector>

namespace sierra {
namespace nalu {

struct HexNElementDescription
{
public:
  static constexpr int dimension = 3;
  static constexpr int numFaces = 6;
  static constexpr int numEdges = 12;
  static constexpr int nodesInBaseElement = 8;
  static constexpr int nodesPerSubElement = 8;

  HexNElementDescription(int p);

  int node_map(int i, int j, int k) const { return nodeMap[i+nodes1D*(j+nodes1D*k)]; };
  int node_map(int i) const { return nodeMap[i]; };
  int bc_node_map(int i, int j) const {return nodeMapBC[i+nodes1D*j]; }
  const std::vector<int>& edge_node_connectivities(int i) const { return edgeNodeConnectivities[i]; }
  const std::vector<int>& face_node_connectivities(int i) const { return faceNodeConnectivities[i]; }
  int volume_node_connectivities(int i) const { return volumeNodeConnectivities[i]; }
  std::array<int, 3> inverse_node_map(int i) const { return inverseNodeMap[i]; }
  const std::vector<int>& side_node_ordinals(int i) const { return sideOrdinalMap[i]; }
  const std::array<int, 8>& sub_element_connectivity(int i) const { return subElementConnectivity[i]; }

  const int polyOrder;
  const int nodes1D;
  const int nodesPerSide;
  const int nodesPerElement;
  const int newNodesPerEdge;
  const int newNodesPerFace;
  const int newNodesPerVolume;
private:
  void set_subelement_connectivity();
  std::vector<int> edge_node_ordinals();
  void set_edge_node_connectivities();
  std::vector<int> face_node_ordinals();
  void set_face_node_connectivities();
  std::vector<int> volume_node_ordinals();
  void set_volume_node_connectivities();
  void set_subelement_connectivites();
  std::pair<int, int> get_edge_offsets(int i, int j, int k, int edge_ordinal);
  std::pair<int, int> get_face_offsets(int i, int j, int k, int face_ordinal);
  void set_base_node_maps();
  void set_tensor_product_node_mappings();
  void set_boundary_node_mappings();
  int& node_map(int i, int j, int k ) { return nodeMap.at(i+nodes1D*(j+nodes1D*k)); };

  std::array<std::vector<int>, numEdges> edgeNodeConnectivities;
  std::array<std::vector<int>, numFaces> faceNodeConnectivities;
  std::vector<int> volumeNodeConnectivities;
  std::vector<std::array<int, nodesInBaseElement>> subElementConnectivity;
  std::vector<int> nodeMap;
  std::vector<std::array<int, dimension>> inverseNodeMap;
  std::vector<int> nodeMapBC;
  std::vector<std::array<int, dimension-1>> inverseNodeMapBC;
  std::array<std::vector<int>, numFaces> sideOrdinalMap;
};

} // namespace nalu
} // namespace Sierra

#endif
