// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PromoteElement_h
#define PromoteElement_h

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <vector>
#include <tuple>
#include <unordered_map>

#include <stk_topology/topology.hpp>
#include <boost/functional/hash/hash.hpp>

namespace stk { namespace mesh { class Part; } }
namespace stk { namespace mesh { class BulkData; } }
namespace stk { namespace mesh { class Selector; } }
namespace stk { namespace mesh { struct Entity; } }
namespace stk { namespace mesh { typedef std::vector<Part*> PartVector; } }
namespace stk { namespace mesh { typedef std::vector<Entity> EntityVector; } }
namespace stk { namespace mesh { typedef std::vector<EntityId> EntityIdVector; } }
namespace sierra { namespace nalu { struct ElementDescription; } }
typedef stk::mesh::Field<double, stk::mesh::Cartesian>  VectorFieldType;

namespace sierra {
namespace nalu {
namespace promotion {

  std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
  promote_elements(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const VectorFieldType& coordField,
    const stk::mesh::PartVector& elemPartsToBePromoted,
    stk::mesh::Part* edgePart,
    stk::mesh::Part* facePart = nullptr);

  stk::mesh::PartVector
  create_boundary_elements(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& meshParts);

} // namespace promotion
} // namespace naluUnit
} // namespace Sierra

namespace sierra {
namespace nalu {
namespace promotion {
namespace internal {

  using ConnectivityMap = std::unordered_map<stk::mesh::Entity, stk::mesh::EntityIdVector>;

  struct EntityIdVectorHash
  {
    std::size_t operator()(const stk::mesh::EntityIdVector& ids) const
    {
      return boost::hash_range(ids.begin(), ids.end());
    }
  };

  using NodesElemMap = std::unordered_map< stk::mesh::EntityIdVector,
                                           stk::mesh::Entity,
                                           EntityIdVectorHash >;

  std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
  promote_elements_quad(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const VectorFieldType& coordField,
    const stk::mesh::PartVector& elemPartsToBePromoted,
    stk::mesh::Part& edgePart);

  std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
  promote_elements_hex(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const VectorFieldType& coordField,
    const stk::mesh::PartVector& elemPartsToBePromoted,
    stk::mesh::Part& edgePart,
    stk::mesh::Part& facePart);

  ConnectivityMap
  connectivity_map_for_parent_rank(
    stk::mesh::BulkData& bulk,
    const int numNewNodesOnTopo,
    const stk::mesh::Selector& selector,
    stk::topology::rank_t parent_rank);

  void add_base_nodes_to_elem_connectivity(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::Entity elem,
    stk::mesh::EntityIdVector& allNodes);

  void add_edge_nodes_to_elem_connectivity(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const ConnectivityMap& edgeConnectivity,
    const stk::mesh::Entity elem,
    stk::mesh::EntityIdVector& allNodes);

  void add_face_nodes_to_elem_connectivity(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const ConnectivityMap& faceConnectivity,
    const stk::mesh::Entity elem,
    stk::mesh::EntityIdVector& allNodes);

  void add_volume_nodes_to_elem_connectivity(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const ConnectivityMap& volumeConnectivity,
    const stk::mesh::Entity elem,
    stk::mesh::EntityIdVector& allNodes);

  void create_nodes_for_connectivity_map(
    stk::mesh::BulkData& bulk,
    const ConnectivityMap& edgeConnectivity);

  stk::mesh::PartVector create_super_elements(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& partsToBePromoted,
    const ConnectivityMap& edgeConnectivity,
    const ConnectivityMap& volumeConnectivity);

  stk::mesh::PartVector create_super_elements(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& partsToBePromoted,
    const ConnectivityMap& edgeConnectivity,
    const ConnectivityMap& faceConnectivity,
    const ConnectivityMap& volumeConnectivity);

  void set_coordinates_quad(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& partsToBePromoted,
    const VectorFieldType& coordField);

  void set_coordinates_hex(
    const stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& partsToBePromoted,
    const VectorFieldType& coordField);

  void perform_parallel_consolidation_of_node_ids(
    const stk::mesh::BulkData& bulk,
    ConnectivityMap& connectivityMap);

  void destroy_entities(
    stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    stk::topology::rank_t rank);

  NodesElemMap
  make_base_nodes_to_elem_map_at_boundary(
    const ElementDescription& desc,
    const stk::mesh::BulkData& mesh,
    const stk::mesh::PartVector& meshParts);

  std::unordered_map<stk::mesh::Entity, stk::mesh::Entity>
  exposed_side_to_super_elem_map(
    const ElementDescription& desc,
    const stk::mesh::BulkData& bulk,
    const stk::mesh::PartVector& base_elem_mesh_parts);

  stk::mesh::PartVector
  create_boundary_elements(
    stk::mesh::BulkData& bulk,
    const ElementDescription& desc,
    const stk::mesh::PartVector& parts);

} // namespace internal
} // namespace promotion
} // namespace naluUnit
} // namespace Sierra

#endif
