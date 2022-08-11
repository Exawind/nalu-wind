// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PromoteElementImpl_h
#define PromoteElementImpl_h

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <vector>
#include <tuple>
#include <unordered_map>

#include <stk_topology/topology.hpp>

namespace stk {
namespace mesh {
class Part;
}
} // namespace stk
namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk
namespace stk {
namespace mesh {
class Selector;
}
} // namespace stk
namespace stk {
namespace mesh {
struct Entity;
}
} // namespace stk
namespace stk {
namespace mesh {
typedef std::vector<Part*> PartVector;
}
} // namespace stk
namespace stk {
namespace mesh {
typedef std::vector<Entity> EntityVector;
}
} // namespace stk
namespace stk {
namespace mesh {
typedef std::vector<EntityId> EntityIdVector;
}
} // namespace stk
namespace sierra {
namespace nalu {
struct ElementDescription;
}
} // namespace sierra
typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;

namespace sierra {
namespace nalu {
struct HexNElementDescription;

namespace impl {

using ConnectivityMap =
  std::unordered_map<stk::mesh::Entity, stk::mesh::EntityIdVector>;

struct EntityIdVectorHash
{
  std::size_t operator()(const stk::mesh::EntityIdVector& ids) const
  {
    if (ids.empty()) {
      return 0;
    }

    constexpr uint32_t shift = 0x9e3779b9;
    auto hash = std::hash<stk::mesh::EntityId>()(ids[0]);
    for (unsigned j = 1; j < ids.size(); ++j) {
      hash ^=
        std::hash<std::size_t>()(ids[j]) + shift + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

using NodesElemMap = std::unordered_map<
  stk::mesh::EntityIdVector,
  stk::mesh::Entity,
  EntityIdVectorHash>;

std::pair<stk::mesh::PartVector, stk::mesh::PartVector> promote_elements_hex(
  std::vector<double> nodeLocs1D,
  stk::mesh::BulkData& bulk,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& elemPartsToBePromoted);

ConnectivityMap connectivity_map_for_parent_rank(
  stk::mesh::BulkData& bulk,
  const int numNewNodesOnTopo,
  const stk::mesh::Selector& selector,
  stk::topology::rank_t parent_rank);

void add_base_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes);

void add_edge_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const ConnectivityMap& edgeConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes);

void add_face_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const ConnectivityMap& faceConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes);

void add_volume_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const ConnectivityMap& volumeConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes);

void create_nodes_for_connectivity_map(
  stk::mesh::BulkData& bulk, const ConnectivityMap& edgeConnectivity);

stk::mesh::PartVector create_super_elements(
  stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const stk::mesh::PartVector& partsToBePromoted,
  const ConnectivityMap& edgeConnectivity,
  const ConnectivityMap& faceConnectivity,
  const ConnectivityMap& volumeConnectivity);

void set_coordinates_hex(
  std::vector<double> nodeLocs1D,
  const stk::mesh::BulkData& bulk,
  const HexNElementDescription& desc,
  const stk::mesh::PartVector& partsToBePromoted,
  const VectorFieldType& coordField);

void perform_parallel_consolidation_of_node_ids(
  const stk::mesh::BulkData& bulk, ConnectivityMap& connectivityMap);

void destroy_entities(
  stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  stk::topology::rank_t rank);

NodesElemMap make_base_nodes_to_elem_map_at_boundary(
  const HexNElementDescription& desc,
  const stk::mesh::BulkData& mesh,
  const stk::mesh::PartVector& meshParts);

std::unordered_map<stk::mesh::Entity, stk::mesh::Entity>
exposed_side_to_super_elem_map(
  const HexNElementDescription& desc,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& base_elem_mesh_parts);

stk::mesh::PartVector create_boundary_elements(
  int p, stk::mesh::BulkData& bulk, const stk::mesh::PartVector& parts);

} // namespace impl
} // namespace nalu
} // namespace sierra
#endif
