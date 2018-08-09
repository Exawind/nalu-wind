/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ElementDescription_h
#define ElementDescription_h

#include <stddef.h>
#include <map>
#include <memory>
#include <vector>

#include <stk_topology/topology.hpp>



namespace sierra {
namespace nalu {

  int poly_order_from_topology(int dimension, stk::topology topo);

  using ordinal_type = int;

struct ElementDescription
{
  /**
   * We implement the high order method using the "super topology" concept in STK.
   *
   * However,"super topologies" in STK only hold information about the number of nodes.
   * An "ElementDescription" provides the remaining topological information
   * as well as the reference coordinates of the nodes.
   */

  using AddedConnectivityOrdinalMap = std::map<ordinal_type, std::vector<ordinal_type>>;
  using AddedNodeLocationsMap  = std::map<ordinal_type, std::vector<double>> ;
  using SubElementConnectivity = std::vector<std::vector<ordinal_type>> ;
public:
  static std::unique_ptr<ElementDescription> create(int dimension, int order);
  static std::unique_ptr<ElementDescription> create(int dimension, stk::topology topo);

  virtual ~ElementDescription();

  ordinal_type node_map(ordinal_type i, ordinal_type j) const { return nodeMap.at(i+nodes1D*j); };
  ordinal_type node_map(ordinal_type i, ordinal_type j, ordinal_type k) const { return nodeMap.at(i+nodes1D*(j+nodes1D*k)); };
  ordinal_type node_map_bc(ordinal_type j) const { return nodeMapBC.at(j); };
  ordinal_type node_map_bc(ordinal_type i, ordinal_type j) const { return nodeMapBC.at(i+j*nodes1D); };

  const ordinal_type* side_node_ordinals(ordinal_type sideOrdinal) const {
    return sideOrdinalMap.at(sideOrdinal).data();
  };

  std::vector<double> nodeLocs1D;

  stk::topology baseTopo;

  int polyOrder;
  int dimension;
  int nodes1D;
  int nodesPerElement;
  int nodesPerSide;
  int numEdges;
  int numFaces;
  int numBoundaries;
  int nodesInBaseElement;
  int nodesPerSubElement;
  int newNodesPerEdge;
  int newNodesPerFace;
  int newNodesPerVolume;

  AddedConnectivityOrdinalMap addedConnectivities;
  AddedConnectivityOrdinalMap edgeNodeConnectivities;
  AddedConnectivityOrdinalMap faceNodeConnectivities;
  AddedConnectivityOrdinalMap volumeNodeConnectivities;
  AddedNodeLocationsMap nodeLocs;
  SubElementConnectivity subElementConnectivity;

  std::vector<ordinal_type> baseNodeOrdinals;
  std::vector<ordinal_type> promotedNodeOrdinals;
  std::vector<ordinal_type> nodeMap;
  std::vector<ordinal_type> nodeMapBC;

  std::vector<std::vector<ordinal_type>> baseEdgeConnectivity;
  std::vector<std::vector<ordinal_type>> baseFaceConnectivity;
  std::vector<std::vector<ordinal_type>> baseFaceEdgeConnectivity;

  std::vector<std::vector<ordinal_type>> inverseNodeMap;
  std::vector<std::vector<ordinal_type>> inverseNodeMapBC;
  std::vector<std::vector<ordinal_type>> faceNodeMap;
  std::vector<std::vector<ordinal_type>> sideOrdinalMap;
protected:
  ElementDescription();
};

} // namespace nalu
} // namespace Sierra

#endif
