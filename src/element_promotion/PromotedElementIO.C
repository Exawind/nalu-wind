/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level nalu      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/PromotedElementIO.h>

#include <element_promotion/PromotedPartHelper.h>
#include <nalu_make_unique.h>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/BulkDataInlinedMethods.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/FieldRestriction.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_topology/topology.tcc>
#include <stk_util/util/ReportHandler.hpp>

#include <Ioss_DBUsage.h>
#include <Ioss_DatabaseIO.h>
#include <Ioss_ElementBlock.h>
#include <Ioss_Field.h>
#include <Ioss_IOFactory.h>
#include <Ioss_NodeBlock.h>
#include <Ioss_Property.h>
#include <Ioss_PropertyManager.h>
#include <Ioss_SideBlock.h>
#include <Ioss_SideSet.h>
#include <Ioss_State.h>
#include "Ionit_Initializer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <element_promotion/ElementDescription.h>

namespace sierra{
namespace nalu{

namespace {
  size_t num_sub_elements(
    const int dim,
    const stk::mesh::BucketVector& buckets,
    unsigned polyOrder)
  {
    ThrowRequire(dim == 2 || dim == 3);
    int faceSize = std::pow(polyOrder,dim-1);
    int elemSize = std::pow(polyOrder,dim);
    auto side_rank = (dim == 2) ? stk::topology::EDGE_RANK : stk::topology::FACE_RANK;
    unsigned numEntities = 0;
    for (const auto* ib : buckets) {
      unsigned subElemsPerElem = (ib->topology().rank() == side_rank) ? faceSize : elemSize;
      numEntities += ib->size()*subElemsPerElem;
    }
    return numEntities;
  }
}

PromotedElementIO::PromotedElementIO(
  const ElementDescription& elem,
  const stk::mesh::MetaData& metaData,
  stk::mesh::BulkData& bulkData,
  const stk::mesh::PartVector& baseParts,
  const std::string& fileName,
  const VectorFieldType& coordField
) : elem_(elem),
    metaData_(metaData),
    bulkData_(bulkData),
    fileName_(fileName),
    coordinates_(coordField),
    nDim_(metaData.spatial_dimension())
{
  Ioss::Init::Initializer init_db;

  Ioss::PropertyManager properties;

  Ioss::Property intSizeAPI("INTEGER_SIZE_API", 8);
  properties.add(intSizeAPI);

  Ioss::Property intSizeDB("INTEGER_SIZE_DB", 8);
  properties.add(intSizeDB);

  databaseIO = Ioss::IOFactory::create(
        "exodus",
        fileName_,
        Ioss::WRITE_RESULTS,
        bulkData_.parallel(),
        properties
  );
  ThrowRequire(databaseIO != nullptr && databaseIO->ok(true));

  output_ = make_unique<Ioss::Region>(databaseIO, "HighOrderOutput"); //sink for databaseIO
  ThrowRequire(output_ != nullptr);

  const stk::mesh::BucketVector& elem_buckets = bulkData_.get_buckets(
    stk::topology::ELEM_RANK, stk::mesh::selectUnion(baseParts));

  size_t numSubElems = num_sub_elements(nDim_, elem_buckets, elem_.polyOrder);
  std::vector<stk::mesh::EntityId> subElemIds;

  bulkData.generate_new_ids(stk::topology::ELEM_RANK,  numSubElems,  subElemIds);
  ThrowRequire(subElemIds.size() == numSubElems);

  superElemParts_ = super_elem_part_vector(baseParts);
  ThrowRequireMsg(part_vector_is_valid_and_nonempty(superElemParts_),
    "Not all element parts have a super-element mirror");

  output_->begin_mode(Ioss::STATE_DEFINE_MODEL);
  write_node_block_definitions(superElemParts_);
  write_elem_block_definitions(superElemParts_);
  write_sideset_definitions(baseParts);
  output_->end_mode(Ioss::STATE_DEFINE_MODEL);

  output_->begin_mode(Ioss::STATE_MODEL);
  write_coordinate_list(superElemParts_);
  write_element_connectivity(superElemParts_, subElemIds);
  write_sideset_connectivity(baseParts);
  output_->end_mode(Ioss::STATE_MODEL);
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_database_data(double currentTime)
{
    output_->begin_mode(Ioss::STATE_TRANSIENT);
    int current_output_step = output_->add_state(currentTime);
    output_->begin_state(current_output_step);

    stk::mesh::BucketVector const& nodeBuckets = bulkData_.get_buckets(
      stk::topology::NODE_RANK, stk::mesh::selectUnion(superElemParts_));

    std::vector<int64_t> ids;
    nodeBlock_->get_field_data("ids", ids);

    for (const auto& pair : fields_) {
      ThrowRequire(pair.second != nullptr);
      const stk::mesh::FieldBase& field = *pair.second;
      if (field.type_is<int>()) {
        put_data_on_node_block<int32_t>(*nodeBlock_, ids, field, nodeBuckets);
      }
      else if (field.type_is<uint32_t>()) {
        put_data_on_node_block<uint32_t>(*nodeBlock_, ids, field, nodeBuckets);
      }
      else if (field.type_is<int64_t>()) {
        put_data_on_node_block<int64_t>(*nodeBlock_, ids, field, nodeBuckets);
      }
      else if (field.type_is<uint64_t>()) {
        put_data_on_node_block<uint64_t>(*nodeBlock_, ids, field, nodeBuckets);
      }
      else if (field.type_is<double>()) {
        put_data_on_node_block<double>(*nodeBlock_, ids, field, nodeBuckets);
      }
      else {
        ThrowRequireMsg(false, "Unsupported type for output");
      }
    }

    output_->end_state(current_output_step);
    output_->end_mode(Ioss::STATE_TRANSIENT);
}
//--------------------------------------------------------------------------
template<typename T> void
PromotedElementIO::put_data_on_node_block(
  Ioss::NodeBlock& nodeBlock,
  const std::vector<int64_t>& ids,
  const stk::mesh::FieldBase& field,
  const stk::mesh::BucketVector& buckets) const
{
  ThrowRequire(field.type_is<T>());
  int fieldLength = field.max_size(stk::topology::NODE_RANK);
  std::vector<T> flatArray(count_entities(buckets)*fieldLength);

  size_t index = 0;
  for (size_t k = 0; k < ids.size(); ++k) {
    stk::mesh::Entity node = bulkData_.get_entity(stk::topology::NODE_RANK, ids[k]);
    const T* field_data = static_cast<T*>(stk::mesh::field_data(field, node));
    for (int j = 0; j < fieldLength; ++j) {
      flatArray[index] = (field_data != nullptr) ? field_data[j] : 0.0;
      ++index;
    }
  }
  nodeBlock.put_field_data(field.name(), flatArray.data(), flatArray.size() * sizeof(T));
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_elem_block_definitions(
  const stk::mesh::PartVector& baseParts)
{
  for(const auto* ip : baseParts) {
    if (ip->topology().rank() == stk::topology::ELEM_RANK) {
      const auto& selector     = *ip & metaData_.locally_owned_part();
      const auto& elemBuckets  = bulkData_.get_buckets(
        stk::topology::ELEM_RANK, selector);
      const size_t numSubElems = num_sub_elements(nDim_,elemBuckets, elem_.polyOrder);
      const auto* baseElemPart = base_elem_part_from_super_elem_part(*ip);

      auto block = make_unique<Ioss::ElementBlock>(
        databaseIO,
        baseElemPart->name(),
        baseElemPart->topology().name(),
        numSubElems
      );
      ThrowRequireMsg(block != nullptr, "Element block creation failed");

      auto result = elementBlockPointers_.insert({ip, block.get()});
      ThrowRequireMsg(result.second, "Attempted to add redundant part");

      output_->add(block.release());
    }
  }
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_node_block_definitions(
  const stk::mesh::PartVector& superElemParts)
{
  const auto& nodeBuckets = bulkData_.get_buckets(
    stk::topology::NODE_RANK, stk::mesh::selectUnion(superElemParts));
  auto nodeCount = count_entities(nodeBuckets);
  auto nodeBlock = make_unique<Ioss::NodeBlock>(
    databaseIO, "nodeblock", nodeCount, nDim_);
  ThrowRequireMsg(nodeBlock != nullptr, "Node block creation failed");
  nodeBlock_ = nodeBlock.get();
  output_->add(nodeBlock.release());
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_sideset_definitions(
  const stk::mesh::PartVector& baseParts)
{
  for(const auto* ip : baseParts) {
    const auto& part = *ip;

    stk::mesh::PartVector subsets = part.subsets();
    if (subsets.empty()) {
      continue;
    }

    auto sideset = make_unique<Ioss::SideSet>(databaseIO, part.name());
    ThrowRequireMsg(sideset != nullptr, "Sideset creation failed");

    for (const auto* subpartPtr : subsets) {
      const auto& subpart = *subpartPtr;

      const auto subpartTopology = subpart.topology();
      if (subpartTopology.rank() != metaData_.side_rank()) {
        continue;
      }

      auto selector = metaData_.locally_owned_part() & subpart;
      const auto& sideBuckets = bulkData_.get_buckets(
        metaData_.side_rank(),
        selector
      );
      const size_t numSubElemsInPart = num_sub_elements(nDim_, sideBuckets, elem_.polyOrder);

      auto block = make_unique<Ioss::SideBlock>(
        databaseIO,
        subpart.name(),
        subpartTopology.name(),
        part.topology().name(),
        numSubElemsInPart
      );
      ThrowRequireMsg(block != nullptr, "Sideblock creation failed");

      auto result = sideBlockPointers_.insert({ subpartPtr, block.get() });
      ThrowRequireMsg(result.second, "Attempted to add redundant subpart");

      sideset->add(block.release());
    }
    output_->add(sideset.release());
  }
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_coordinate_list(const stk::mesh::PartVector& superElemParts)
{
  const auto& nodeBuckets =
      bulkData_.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(superElemParts));

  auto nodeCount = count_entities(nodeBuckets);

  std::vector<int64_t> node_ids;
  std::vector<double> coordvec;
  node_ids.reserve(nodeCount);
  coordvec.reserve(nodeCount*nDim_);

  size_t numCoords = nodeCount * nDim_;

  for (const auto* ib : nodeBuckets) {
    const stk::mesh::Bucket& b = *ib;
    const stk::mesh::Bucket::size_type length = b.size();
    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const auto& node = b[k];
      node_ids.push_back(bulkData_.identifier(node));
      const double* coords = stk::mesh::field_data(coordinates_,node);

      for (unsigned j = 0; j < nDim_; ++j) {
        coordvec.push_back(coords[j]);
      }
    }
  }
  nodeBlock_->put_field_data("ids", node_ids);
  nodeBlock_->put_field_data("mesh_model_coordinates", coordvec.data(), numCoords * sizeof(double));
}
//--------------------------------------------------------------------------
void
PromotedElementIO::write_element_connectivity(
  const stk::mesh::PartVector& baseParts,
  const std::vector<stk::mesh::EntityId>& entityIds)
{
  for(const auto* ip : baseParts) {
    const stk::mesh::Part& part = *ip;
    if(part.topology().rank() !=stk::topology::ELEM_RANK) {
      continue;
    }

    const auto& selector = metaData_.locally_owned_part() & part;
    const auto& elemBuckets = bulkData_.get_buckets(stk::topology::ELEM_RANK, selector);

    const size_t numSubElementsInBlock = num_sub_elements(nDim_, elemBuckets, elem_.polyOrder);
    const unsigned nodesPerLinearElem = elem_.nodesPerSubElement;
    std::vector<int64_t> connectivity(nodesPerLinearElem*numSubElementsInBlock);
    std::vector<int64_t> globalSubElementIds(numSubElementsInBlock);

    int connIndex = 0;
    unsigned subElementCounter = 0;
    for (const auto* ib: elemBuckets) {
      const stk::mesh::Bucket& b = *ib;
      const auto length = b.size();
      for (size_t k = 0; k < length; ++k) {
        const auto* node_rels = b.begin_nodes(k);
        const auto& subElems = elem_.subElementConnectivity;
        const auto numberSubElements = subElems.size();

        for (unsigned subElementIndex = 0; subElementIndex < numberSubElements; ++subElementIndex) {
          globalSubElementIds.at(subElementCounter) = entityIds[subElementCounter];

          const auto& localIndices = subElems.at(subElementIndex);
          for (unsigned j = 0; j < nodesPerLinearElem; ++j) {
            connectivity[connIndex] = bulkData_.identifier(node_rels[localIndices[j]]);
            ++connIndex;
          }
          ++subElementCounter;
        }
      }
    }
    elementBlockPointers_.at(ip)->put_field_data("ids",  globalSubElementIds);
    elementBlockPointers_.at(ip)->put_field_data("connectivity", connectivity);
  }
}

void
PromotedElementIO::write_sideset_connectivity(
  const stk::mesh::PartVector&  /*baseParts*/)
{
  //FIXME(rcknaus): implement
}
//--------------------------------------------------------------------------
void
PromotedElementIO::add_fields(const std::vector<stk::mesh::FieldBase*>& fields)
{
  output_->begin_mode(Ioss::STATE_DEFINE_TRANSIENT);
  for (const auto* fieldPtr : fields) {
    if (fieldPtr == nullptr) {
      continue;
    }
    const auto& field = *fieldPtr;

    auto result = fields_.insert({fieldPtr->name(),fieldPtr});
    bool wasFieldAdded = result.second;

    if (wasFieldAdded) {
      int nb_size = nodeBlock_->get_property("entity_count").get_int();

      auto iossType = Ioss::Field::DOUBLE;
      if (field.type_is<uint32_t>() || field.type_is<int32_t>()) {
        iossType = Ioss::Field::INT32;
      }
      else if (field.type_is<uint64_t>() || field.type_is<int64_t>()) {
       iossType = Ioss::Field::INT64;
      }
      else {
        ThrowRequireMsg(field.type_is<double>(), "Only (u)int32, (u)int64, and double fields supported");
      }

      nodeBlock_->field_add(
        Ioss::Field(
          field.name(),
          iossType,
          storage_name(field),
          Ioss::Field::TRANSIENT,
          nb_size
        )
      );
    }
  }
  output_->end_mode(Ioss::STATE_DEFINE_TRANSIENT);
}
//-----------------------------------------------------------------------
std::string
PromotedElementIO::storage_name(const stk::mesh::FieldBase& field) const
{
  std::string storageType;
  switch (field.max_size(stk::topology::NODE_RANK)) {
    case 1:
    {
      storageType = "scalar";
      break;
    }
    case 2:
    {
      storageType = "vector_2d";
      break;
    }
    case 3:
    {
      storageType = "vector_3d";
      break;
    }
    case 4:
    {
      storageType = "full_tensor_22";
      break;
    }
    case 6:
    {
      storageType = "sym_tensor_33";
      break;
    }
    case 9:
    {
      storageType = "full_tensor_36";
      break;
    }
    default: {
      storageType = "GeneralField";
      break;
    }
  }
  return storageType;
}

}  // namespace nalu
}  // namespace sierra
