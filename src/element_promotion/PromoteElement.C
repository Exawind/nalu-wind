/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <element_promotion/PromoteElement.h>
#include <element_promotion/PromotedPartHelper.h>
#include <element_promotion/ElementDescription.h>
#include <master_element/Hex8CVFEM.h>
#include <master_element/MasterElement.h>
#include <master_element/Quad42DCVFEM.h>  
#include <NaluEnv.h>
#include <BucketLoop.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/HashEntityAndEntityKey.hpp>
#include <stk_mesh/base/CreateEdges.hpp>
#include <stk_mesh/base/CreateFaces.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_util/parallel/CommSparse.hpp>
#include <stk_util/parallel/ParallelComm.hpp>
#include <stk_util/util/ReportHandler.hpp>

#include <boost/functional/hash/hash.hpp>

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <limits>

namespace sierra {
namespace nalu {
namespace promotion {

std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
promote_elements(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& partsToBePromoted,
  stk::mesh::Part* edgePart,
  stk::mesh::Part* facePart)
{
  // only quads and hexs implemented

  if (desc.dimension == 2) {
    ThrowRequire(edgePart != nullptr);
    return internal::promote_elements_quad(bulk, desc, coordField, partsToBePromoted, *edgePart);
  }
  ThrowRequire(facePart != nullptr);
  return internal::promote_elements_hex(bulk, desc, coordField, partsToBePromoted, *edgePart, *facePart);
}

stk::mesh::PartVector
create_boundary_elements(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& parts)
{
  return internal::create_boundary_elements(bulk, desc, parts);
}

} // namespace promotion
} // namespace naluUnit
} // namespace Sierra
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
namespace sierra {
namespace nalu {
namespace promotion {
namespace internal {

std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
promote_elements_quad(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& partsToBePromoted,
  stk::mesh::Part& edgePart)
{
  ThrowRequire(check_parts_for_promotion(partsToBePromoted));
  ThrowRequireMsg(!bulk.is_automatic_aura_on(), "Promotion not yet tested for automatic aura");
  ThrowRequire(desc.baseTopo == stk::topology::QUAD_4_2D);

  stk::mesh::Selector edgeSelector = edgePart | stk::mesh::selectUnion(base_edge_parts(partsToBePromoted));
  stk::mesh::Selector volSelector = stk::mesh::selectUnion(base_elem_parts(partsToBePromoted));

  stk::mesh::create_edges(bulk, volSelector, &edgePart);

  bulk.modification_begin();

  ConnectivityMap edgeNodeMap = connectivity_map_for_parent_rank(
    bulk,
    desc.newNodesPerEdge,
    edgeSelector,
    stk::topology::EDGE_RANK
  );

  ConnectivityMap volNodeMap = connectivity_map_for_parent_rank(
    bulk,
    desc.newNodesPerVolume,
    volSelector,
    stk::topology::ELEM_RANK
  );

  stk::mesh::PartVector promotedElemParts
     = create_super_elements(bulk, desc, partsToBePromoted, edgeNodeMap, volNodeMap);

  destroy_entities(bulk, edgePart, stk::topology::EDGE_RANK);
  bulk.modification_end();

  stk::mesh::PartVector promotedSideParts = create_boundary_elements(bulk, desc, partsToBePromoted);

  set_coordinates_quad(bulk, desc, promotedElemParts, coordField);

  return std::make_pair(promotedElemParts, promotedSideParts);
}
//--------------------------------------------------------------------------
std::pair<stk::mesh::PartVector, stk::mesh::PartVector>
promote_elements_hex(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const VectorFieldType& coordField,
  const stk::mesh::PartVector& partsToBePromoted,
  stk::mesh::Part& edgePart,
  stk::mesh::Part&  /* facePart */)
{
  ThrowRequire(check_parts_for_promotion(partsToBePromoted));
  ThrowRequireMsg(!bulk.is_automatic_aura_on(), "Promotion not yet tested for automatic aura");
  ThrowRequire(desc.baseTopo == stk::topology::HEX_8);

  stk::mesh::Selector edgeSelector = stk::mesh::selectUnion(base_edge_parts(partsToBePromoted));
  stk::mesh::Selector faceSelector = stk::mesh::selectUnion(base_face_parts(partsToBePromoted));
  stk::mesh::Selector volSelector  = stk::mesh::selectUnion(base_elem_parts(partsToBePromoted));

  stk::mesh::create_edges(bulk, volSelector, &edgePart);
  stk::mesh::create_faces(bulk, volSelector);

  stk::mesh::Selector allEdgeSelector = edgePart | edgeSelector;
  stk::mesh::Selector allFaceSelector = bulk.mesh_meta_data().get_topology_root_part(stk::topology::QUAD_4);
  stk::mesh::Selector newFaceSelector = (!faceSelector) & allFaceSelector;

  bulk.modification_begin();

  ConnectivityMap edgeNodeMap = connectivity_map_for_parent_rank(
    bulk,
    desc.newNodesPerEdge,
    allEdgeSelector,
    stk::topology::EDGE_RANK
  );

  ConnectivityMap faceNodeMap = connectivity_map_for_parent_rank(
    bulk,
    desc.newNodesPerFace,
    allFaceSelector,
    stk::topology::FACE_RANK
  );

  ConnectivityMap volNodeMap = connectivity_map_for_parent_rank(
    bulk,
    desc.newNodesPerVolume,
    volSelector,
    stk::topology::ELEM_RANK
  );

  stk::mesh::PartVector promotedElemParts = create_super_elements(
    bulk,
    desc,
    partsToBePromoted,
    edgeNodeMap,
    faceNodeMap,
    volNodeMap
  );

  destroy_entities(bulk, edgePart, stk::topology::EDGE_RANK);
  destroy_entities(bulk, newFaceSelector, stk::topology::FACE_RANK);

  bulk.modification_end();

  stk::mesh::PartVector promotedSideParts = create_boundary_elements(bulk, desc, partsToBePromoted);

  set_coordinates_hex(bulk, desc, promotedElemParts, coordField);

  return std::make_pair(promotedElemParts, promotedSideParts);
}
//--------------------------------------------------------------------------
template<class LOOP_BODY>
void bucket_loop(const stk::mesh::BucketVector& buckets, LOOP_BODY inner_loop_body)
{
  for(const stk::mesh::Bucket* bptr : buckets) {
    const stk::mesh::Bucket& bkt = *bptr;
    for(size_t j=0; j<bkt.size(); ++j)  {
      inner_loop_body(bkt[j]);
    }
  }
}
//--------------------------------------------------------------------------
stk::mesh::PartVector
create_super_elements(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& elemPartsToBePromoted,
  const ConnectivityMap& edgeConnectivity,
  const ConnectivityMap& volumeConnectivity)
{
  auto selector = stk::mesh::selectUnion(elemPartsToBePromoted);
  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  stk::mesh::EntityIdVector elemIds;
  bulk.generate_new_ids(stk::topology::ELEM_RANK, count_entities(elem_buckets), elemIds);

  stk::mesh::EntityIdVector elemConnectivity(desc.nodesPerElement, 0);

  stk::mesh::PartVector promotedElemParts;
  size_t idCounter = 0;
  for (auto* ip : elemPartsToBePromoted) {
    auto& superPart = *super_elem_part(*ip);

    const auto& elem_part_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, *ip);
    bucket_loop(elem_part_buckets, [&](const stk::mesh::Entity elem) {
      add_base_nodes_to_elem_connectivity(bulk, desc, elem, elemConnectivity);
      add_edge_nodes_to_elem_connectivity(bulk, desc, edgeConnectivity, elem, elemConnectivity);
      add_volume_nodes_to_elem_connectivity(bulk, desc, volumeConnectivity, elem, elemConnectivity);

      stk::mesh::declare_element(bulk, superPart, elemIds[idCounter], elemConnectivity);
      ++idCounter;
    });
    promotedElemParts.push_back(&superPart);
  }

  return promotedElemParts;
}
//--------------------------------------------------------------------------
stk::mesh::PartVector
create_super_elements(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& elemPartsToBePromoted,
  const ConnectivityMap& edgeConnectivity,
  const ConnectivityMap& faceConnectivity,
  const ConnectivityMap& volumeConnectivity)
{
  auto selector = stk::mesh::selectUnion(elemPartsToBePromoted);
  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  stk::mesh::EntityIdVector elemIds;
  bulk.generate_new_ids(stk::topology::ELEM_RANK, count_entities(elem_buckets), elemIds);

  stk::mesh::EntityIdVector elemConnectivity(desc.nodesPerElement, 0);

  stk::mesh::PartVector promotedElemParts;
  size_t idCounter = 0;
  for (auto* ip : elemPartsToBePromoted) {
    auto& superPart = *super_elem_part(*ip);

    const auto& elem_part_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, *ip);
    bucket_loop(elem_part_buckets, [&](const stk::mesh::Entity elem) {
      add_base_nodes_to_elem_connectivity(bulk, desc, elem, elemConnectivity);
      add_edge_nodes_to_elem_connectivity(bulk, desc, edgeConnectivity, elem, elemConnectivity);
      add_face_nodes_to_elem_connectivity(bulk, desc, faceConnectivity, elem, elemConnectivity);
      add_volume_nodes_to_elem_connectivity(bulk, desc, volumeConnectivity, elem, elemConnectivity);

      stk::mesh::declare_element(bulk, superPart, elemIds[idCounter], elemConnectivity);
      ++idCounter;
    });
    promotedElemParts.push_back(&superPart);
  }

  return promotedElemParts;
}
//--------------------------------------------------------------------------
stk::mesh::PartVector
create_boundary_elements(
  stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& parts)
{
  auto sideToSuperElemMap = exposed_side_to_super_elem_map(desc, bulk, parts);

  auto side_rank = bulk.mesh_meta_data().side_rank();
  const auto& side_buckets = bulk.get_buckets(side_rank, stk::mesh::selectUnion(parts));
  const auto numNewFace = count_entities(side_buckets);

  std::vector<stk::mesh::EntityId> availableFaceIds(numNewFace);
  bulk.generate_new_ids(side_rank, numNewFace, availableFaceIds);

  stk::mesh::PartVector soloFacePart(1, nullptr);
  stk::mesh::PartVector superSideParts;

  size_t faceIdIndex = 0;

  bulk.modification_begin();
  for (const auto* ipart : parts) {
    for (const auto* subset : ipart->subsets()) {
      if ( subset->topology().rank() == side_rank && !subset->topology().is_super_topology()) {
        soloFacePart[0] = super_subset_part(*subset);
        ThrowRequire(soloFacePart[0]  != nullptr);

        const auto& buckets = bulk.get_buckets(side_rank, *subset);
        bucket_loop(buckets, [&](const stk::mesh::Entity side)  {
          stk::mesh::Entity superSide = bulk.declare_solo_side(availableFaceIds[faceIdIndex], soloFacePart);
          stk::mesh::Entity superElem = sideToSuperElemMap.at(side);

          ThrowRequireMsg(bulk.num_elements(side) == 1u, "Multiple elements attached to boundary side");
          const auto sideOrdinal = bulk.begin_element_ordinals(side)[0];
          const stk::mesh::Entity* elem_node_rels = bulk.begin_nodes(superElem);
          const auto& sideNodeOrdinals = desc.side_node_ordinals(sideOrdinal);

          for (int j = 0; j < desc.nodesPerSide; ++j) {
            bulk.declare_relation(superSide, elem_node_rels[sideNodeOrdinals[j]], j);
          }
          bulk.declare_relation(superElem, superSide, sideOrdinal);

          ++faceIdIndex;
        });
      }
      superSideParts.push_back(soloFacePart[0]);
    }
  }
  bulk.modification_end();
  return superSideParts;
}
//--------------------------------------------------------------------------
stk::mesh::EntityId
choose_consistent_node_id(
  stk::mesh::EntityId myId,
  stk::mesh::EntityId theirId)
{
  // parallel consistency rule is to choose the lowest id
  return (myId < theirId) ? myId : theirId;
}
//--------------------------------------------------------------------------
void
perform_parallel_consolidation_of_node_ids(
  const stk::mesh::BulkData& bulk,
  ConnectivityMap& connectivityMap)
{
  if (bulk.parallel_size() == 1 || connectivityMap.empty()) {
    return; // unnecessary for serial / empty maps
  }

  stk::topology::rank_t domainTopoRank = bulk.entity_rank(connectivityMap.begin()->first);
  if (domainTopoRank == stk::topology::ELEM_RANK) {
    return; // elem rank ids are parallel-consistent already
  }

  struct EntityNodeSharing {
    stk::mesh::EntityId owningId;
    int localIndex;
    stk::mesh::EntityId nodeId;
  };

  stk::CommSparse comm_spec(bulk.parallel());
  stk::pack_and_communicate(comm_spec, [&]() {
    for (const auto& pair : connectivityMap) {
      auto entKey = bulk.entity_key(pair.first);
      const auto& nodeIds = pair.second;
      ThrowRequire(entKey.rank() == domainTopoRank);

      std::vector<int> procs;
      bulk.comm_shared_procs(entKey, procs);
      for (int otherProcRank : procs) {
        if (otherProcRank != bulk.parallel_rank()) {
          for (unsigned localIndex = 0; localIndex < nodeIds.size(); ++localIndex) {
            EntityNodeSharing ensh;
            ensh.owningId = entKey.id();
            ensh.localIndex = localIndex;
            ensh.nodeId = nodeIds.at(localIndex);
            comm_spec.send_buffer(otherProcRank).pack(ensh);
          }
        }
      }
    }
  });

  stk::unpack_communications(comm_spec, [&](int otherProcRank) {
    EntityNodeSharing ensh;
    comm_spec.recv_buffer(otherProcRank).unpack(ensh);
    stk::mesh::Entity entity = bulk.get_entity(domainTopoRank, ensh.owningId);
    stk::mesh::EntityId theirId = ensh.nodeId;

    stk::mesh::EntityId* myId = &connectivityMap.at(entity).at(ensh.localIndex);
    *myId = choose_consistent_node_id(*myId, theirId);
  });
}
//--------------------------------------------------------------------------
void
create_nodes_for_connectivity_map(
  stk::mesh::BulkData& bulk,
  const ConnectivityMap& map)
{
  // Rule: new node inherits the parallel ownership rule of its parent topology, e.g.
  // a "edge node" is owned by the same process that owns the edge its on.

  for (const auto& pair : map) {
    std::vector<int> procs;
    bulk.comm_shared_procs(bulk.entity_key(pair.first), procs);
    for (stk::mesh::EntityId id : pair.second) {
      stk::mesh::Entity node = bulk.declare_entity(stk::topology::NODE_RANK, id, stk::mesh::PartVector{});
      for (int proc : procs) {
        if (proc != bulk.parallel_rank()) {
          bulk.add_node_sharing(node, proc);
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
ConnectivityMap
connectivity_map_for_parent_rank(
  stk::mesh::BulkData& bulk,
  const int numNewNodesOnTopo,
  const stk::mesh::Selector& selector,
  stk::topology::rank_t parent_rank)
{
  const auto& buckets = bulk.get_buckets(parent_rank, selector);
  size_t numNewNodes = count_entities(buckets) * numNewNodesOnTopo;

  stk::mesh::EntityIdVector newNodeIds;
  bulk.generate_new_ids(stk::topology::NODE_RANK, numNewNodes, newNodeIds);

  ConnectivityMap map;
  auto beginIterator = newNodeIds.begin();
  bucket_loop(buckets, [&](stk::mesh::Entity entity) {
    auto endIterator = beginIterator + numNewNodesOnTopo;
    map.insert({entity, stk::mesh::EntityIdVector{beginIterator, endIterator}});
    beginIterator = endIterator;
  });

  perform_parallel_consolidation_of_node_ids(bulk, map);
  create_nodes_for_connectivity_map(bulk, map);
  return map;
}
//--------------------------------------------------------------------------
void
add_base_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes)
{
  const auto* base_elem_rels = bulk.begin_nodes(elem);
  for (int j = 0; j < desc.nodesInBaseElement; ++j) {
    allNodes[j] = bulk.identifier(base_elem_rels[desc.baseNodeOrdinals.at(j)]);
  }
}
//--------------------------------------------------------------------------
int index_edge_nodes(int i, int len, stk::mesh::Permutation perm)
{
  // only two permutations: (0,1) and (1,0)
  if (perm != stk::mesh::Permutation::DEFAULT_PERMUTATION) {
    return len-i-1;
  }
  return i;
}
//--------------------------------------------------------------------------
int index_face_nodes(
  int i,
  int j,
  int len1D,
  stk::mesh::Permutation perm)
{
  int ix = -1;
  int iy = -1;

  // "reversed" indices
  int ir = len1D-1-i;
  int jr = len1D-1-j;

  // map the new nodes consistently with the permutation
  switch (static_cast<int>(perm))
  {
    case 0: // 0, 1, 2, 3
    {
      ix = i;
      iy = j;
      break;
    }
    case 1: // 3, 0, 1, 2
    {
      ix = jr;
      iy = i;
      break;
    }
    case 2: // 2, 3, 0, 1
    {
      ix = ir;
      iy = jr;
      break;
    }
    case 3: // 1, 2, 3, 0
    {
      ix = j;
      iy = ir;
      break;
    }
    case 4: // 0, 3, 2, 1
    {
      ix = j;
      iy = i;
      break;
    }
    case 5: // 3, 2, 1, 0
    {
      ix = i;
      iy = jr;
      break;
    }
    case 6: //2, 1, 0, 3
    {
      ix = jr;
      iy = ir;
      break;
    }
    case 7: //1, 0, 3, 2
    {
      ix = ir;
      iy = j;
      break;
    }
    default: {
      ThrowRequireMsg(false, "Invalid permutation of quad face");
    }
  }

  return ix + iy * len1D;
}
//--------------------------------------------------------------------------
void
add_edge_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const ConnectivityMap& edgeConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes)
{
  const auto* edge_rels = bulk.begin_edges(elem);
  const auto* edge_ords = bulk.begin_edge_ordinals(elem);
  const auto* perm = bulk.begin_edge_permutations(elem);
  int newNodesPerEdge = desc.newNodesPerEdge;
  for (unsigned edge_index = 0; edge_index < bulk.num_edges(elem); ++edge_index) {
    int edge_ord = edge_ords[edge_index];
    const stk::mesh::EntityIdVector& nodeIds = edgeConnectivity.at(edge_rels[edge_ord]);
    const std::vector<int>& ords = desc.edgeNodeConnectivities.at(edge_ord);

    ThrowAssert(nodeIds.size() == ords.size());
    ThrowAssert(static_cast<int>(ords.size()) == newNodesPerEdge);
    for (int i = 0; i < newNodesPerEdge; ++i) {
      allNodes.at(ords.at(i)) = nodeIds.at(index_edge_nodes(i, newNodesPerEdge, perm[edge_ord]));
    }
  }
}
//--------------------------------------------------------------------------
void
add_face_nodes_to_elem_connectivity(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const ConnectivityMap& faceConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes)
{
  const auto* face_rels = bulk.begin_faces(elem);
  const auto* face_ords = bulk.begin_edge_ordinals(elem);
  const auto* face_perm = bulk.begin_face_permutations(elem);
  int newNodesPerEdge = desc.newNodesPerEdge;
  for (unsigned face_index = 0; face_index < bulk.num_faces(elem); ++face_index) {
    int face_ord = face_ords[face_index];
    const stk::mesh::EntityIdVector& nodeIds = faceConnectivity.at(face_rels[face_ord]);
    const std::vector<int>& ords = desc.faceNodeConnectivities.at(face_index);

    ThrowAssert(nodeIds.size() == ords.size());
    ThrowAssert(desc.newNodesPerFace == static_cast<int>(ords.size()));
    ThrowAssert(desc.newNodesPerFace == newNodesPerEdge * newNodesPerEdge);

    for (int j = 0; j < newNodesPerEdge; ++j) {
      for (int i = 0; i < newNodesPerEdge; ++i) {
        allNodes.at(ords.at(i + j * newNodesPerEdge)) =
            nodeIds.at(index_face_nodes(i, j, newNodesPerEdge, face_perm[face_ord]));
      }
    }
  }
}
//--------------------------------------------------------------------------
void
add_volume_nodes_to_elem_connectivity(
  const stk::mesh::BulkData&  /* bulk */,
  const ElementDescription& desc,
  const ConnectivityMap& volumeConnectivity,
  const stk::mesh::Entity elem,
  stk::mesh::EntityIdVector& allNodes)
{
  const auto& nodes = volumeConnectivity.at(elem);
  const auto& ords = desc.volumeNodeConnectivities.at(0);
  ThrowAssert(nodes.size() == ords.size());

  for (unsigned j = 0; j < ords.size(); ++j) {
    allNodes.at(ords.at(j)) = nodes.at(j);
  }
}
//--------------------------------------------------------------------------
bool destroy_entity(
  stk::mesh::BulkData& bulk,
  stk::mesh::Entity entity)
{
  stk::mesh::EntityRank entityRank = bulk.entity_rank(entity);

  const auto highestEntityRank =
      static_cast<stk::mesh::EntityRank>(bulk.mesh_meta_data().entity_rank_count()-1);

  for (auto irank = highestEntityRank; irank != entityRank; --irank) {
    // copy the data
    auto relatives = stk::mesh::EntityVector{bulk.begin(entity, irank), bulk.end(entity,irank)};
    auto relative_ordinals = std::vector<stk::mesh::ConnectivityOrdinal>{
      bulk.begin_ordinals(entity, irank),bulk.end_ordinals(entity, irank)
    };

    for (unsigned irel = 0; irel < relatives.size(); ++irel) {
      bool del = bulk.destroy_relation( relatives[irel], entity, relative_ordinals[irel]);
      ThrowRequireMsg(del, "Failed to disconnect entity: " + std::to_string(bulk.identifier(entity)));
    }
  }
  return bulk.destroy_entity(entity);
}
//--------------------------------------------------------------------------
void
destroy_entities(
  stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  stk::topology::rank_t rank)
{
  stk::mesh::EntityVector entities;
  stk::mesh::get_selected_entities(selector, bulk.get_buckets(rank, selector), entities);

  for (stk::mesh::Entity entity : entities) {
    ThrowRequire(bulk.is_valid(entity));
    if (bulk.bucket(entity).owned()) {
      bool destroyed = destroy_entity(bulk, entity);
      ThrowRequireMsg(destroyed, "Failed to destroy entity: " + std::to_string(bulk.identifier(entity)));
    }
  }
}
//--------------------------------------------------------------------------
void
set_coordinates_quad(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& promotedPartVector,
  const VectorFieldType& coordField)
{
  Quad42DSCS meQuad;
  auto selector = stk::mesh::selectUnion(promotedPartVector);
  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);

  int baseNodes = desc.nodesInBaseElement;
  std::vector<double> baseCoords(baseNodes*desc.dimension);
  std::vector<double> physCoords(desc.dimension);

  bucket_loop(elem_buckets, [&](const stk::mesh::Entity elem) {
    ThrowAssert(desc.nodesPerElement == static_cast<int>(bulk.num_nodes(elem)));
    const stk::mesh::Entity* node_rels = bulk.begin_nodes(elem);

    for (int ord : desc.baseNodeOrdinals) {
      double* coords = stk::mesh::field_data(coordField, node_rels[ord]);
      for (int d = 0; d < desc.dimension; ++d) {
        baseCoords[d * 4 + ord] = coords[d];
      }
    }

    for (int ord : desc.promotedNodeOrdinals) {
      const auto& isoParCoords = desc.nodeLocs.at(ord);
      meQuad.interpolatePoint(desc.dimension, isoParCoords.data(), baseCoords.data(), physCoords.data());
      double* coords = stk::mesh::field_data(coordField, node_rels[ord]);
      for (int d = 0; d < desc.dimension; ++d) {
        coords[d] = physCoords[d];
      }
    }
  });
}
//--------------------------------------------------------------------------
void
set_coordinates_hex(
  const stk::mesh::BulkData& bulk,
  const ElementDescription& desc,
  const stk::mesh::PartVector& promotedPartVector,
  const VectorFieldType& coordField)
{
  HexSCS meHex;
  auto selector = stk::mesh::selectUnion(promotedPartVector);
  const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK,
    selector);

  int baseNodes = desc.nodesInBaseElement;
  std::vector<double> baseCoords(baseNodes * desc.dimension);
  std::vector<double> physCoords(desc.dimension);

  bucket_loop(elem_buckets,  [&](const stk::mesh::Entity elem) {
    ThrowAssert(desc.nodesPerElement == static_cast<int>(bulk.num_nodes(elem)));
    const stk::mesh::Entity* node_rels = bulk.begin_nodes(elem);

    int baseIndex = 0;
    for (int ord : desc.baseNodeOrdinals) {
      double* coords = stk::mesh::field_data(coordField, node_rels[ord]);
      for (int d = 0; d < desc.dimension; ++d) {
        baseCoords[d * 8 + baseIndex] = coords[d];
      }
      ++baseIndex;
    }

    for (int ord : desc.promotedNodeOrdinals) {
      auto isoParCoords = desc.nodeLocs.at(ord);
      meHex.interpolatePoint(
        desc.dimension,
        isoParCoords.data(),
        baseCoords.data(),
        physCoords.data()
      );
      double* coords = stk::mesh::field_data(coordField, node_rels[ord]);

      for (int d = 0; d < desc.dimension; ++d) {
        coords[d] = physCoords[d];
      }
    }
  });
}
//--------------------------------------------------------------------------
NodesElemMap
make_base_nodes_to_elem_map_at_boundary(
  const ElementDescription& desc,
  const stk::mesh::BulkData& mesh,
  const stk::mesh::PartVector& meshParts)
{
  /* For elements connected to a face, the method
   * generates a map between a (sorted) vector of the element's
   * node ids to the element itself
   */
  const auto& baseElemSideBuckets = mesh.get_buckets(
    mesh.mesh_meta_data().side_rank(),
    stk::mesh::selectUnion(meshParts)
  );

  const auto baseNumNodes = desc.nodesInBaseElement;
  NodesElemMap nodesToElemMap;
  stk::mesh::EntityIdVector parents(baseNumNodes);
  bucket_loop (baseElemSideBuckets, [&](stk::mesh::Entity side) {
    ThrowRequireMsg(mesh.num_elements(side) == 1u, "Multiple elements attached to boundary side");
    const stk::mesh::Entity parent_elem = mesh.begin_elements(side)[0];

    const auto* node_rels = mesh.begin_nodes(parent_elem);
    ThrowAssert(mesh.num_nodes(parent_elem) == parents.size());

    for (int j = 0; j < desc.nodesInBaseElement; ++j) {
      parents.at(j) = mesh.identifier(node_rels[j]);
    }
    std::sort(parents.begin(), parents.end());
    nodesToElemMap.insert({parents, parent_elem});
  });
  return nodesToElemMap;
}
//--------------------------------------------------------------------------
std::unordered_map<stk::mesh::Entity, stk::mesh::Entity>
exposed_side_to_super_elem_map(
  const ElementDescription& desc,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::PartVector& base_elem_mesh_parts)
{
  /*
   * Generates a map between each exposed face and the super-element
   * notionally attached to that exposed face.
   */
  ThrowRequire(part_vector_is_valid_and_nonempty(super_elem_part_vector(base_elem_mesh_parts)));

  const auto& superElemBuckets = bulk.get_buckets(
    stk::topology::ELEM_RANK,
    stk::mesh::selectUnion(super_elem_part_vector(base_elem_mesh_parts))
  );
  auto nodesToElemMap =
      make_base_nodes_to_elem_map_at_boundary(desc, bulk, base_elem_mesh_parts);

  std::unordered_map<stk::mesh::Entity, stk::mesh::Entity> elemToSuperElemMap;
  elemToSuperElemMap.reserve(nodesToElemMap.size());

  const auto baseNumNodes = desc.nodesInBaseElement;
  std::vector<stk::mesh::EntityId> parents(baseNumNodes);

  bucket_loop(superElemBuckets, [&](stk::mesh::Entity superElem) {
    const auto* node_rels = bulk.begin_nodes(superElem);
    ThrowAssert(static_cast<int>(bulk.num_nodes(superElem)) > baseNumNodes || desc.polyOrder == 1);

    // Requires the convention that the base nodes are stored
    // first in the elem node relations still holds
    for (int j = 0; j < baseNumNodes ; ++j) {
      parents.at(j) = bulk.identifier(node_rels[desc.baseNodeOrdinals.at(j)]);
    }
    std::sort(parents.begin(), parents.end());

    auto it = nodesToElemMap.find(parents);
    if (it != nodesToElemMap.end()) {
      const stk::mesh::Entity baseElem = it->second;
      auto result = elemToSuperElemMap.insert({baseElem, superElem});
      ThrowRequireMsg(result.second,
        "Multiple superElems with same parent nodes as the base elements found");
    }
  });
  nodesToElemMap.clear();

  const stk::mesh::BucketVector& boundary_buckets = bulk.get_buckets(
    bulk.mesh_meta_data().side_rank(), stk::mesh::selectUnion(base_elem_mesh_parts)
  );

  std::unordered_map<stk::mesh::Entity, stk::mesh::Entity> exposedSideToSuperElemMap;
  exposedSideToSuperElemMap.reserve(elemToSuperElemMap.size());

  bucket_loop(boundary_buckets, [&](stk::mesh::Entity side){;
    ThrowAssert(bulk.num_elements(side) == 1u);
    const stk::mesh::Entity baseElem = bulk.begin_elements(side)[0];
    const stk::mesh::Entity superElem = elemToSuperElemMap.at(baseElem);
    auto result = exposedSideToSuperElemMap.insert({side, superElem});
    ThrowRequireMsg(result.second,
      "Multiple super elements associated with the same face");
  });

  return exposedSideToSuperElemMap;
}

} // namespace internal
} // namespace promotion
} // namespace naluUnit
} // namespace Sierra
