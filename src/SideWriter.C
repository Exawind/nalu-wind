// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "SideWriter.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/util/ReportHandler.hpp>

#include <Ioss_Region.h>
#include <Ioss_DBUsage.h>
#include <Ioss_DatabaseIO.h>
#include <Ioss_ElementBlock.h>
#include <Ioss_Field.h>
#include <Ioss_IOFactory.h>
#include <Ioss_NodeBlock.h>
#include <Ioss_Property.h>
#include <Ioss_PropertyManager.h>
#include <Ioss_State.h>
#include "Ionit_Initializer.h"

#include "NaluParsing.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace sierra {
namespace nalu {

namespace {

int
count_ent(const stk::mesh::BucketVector& buckets)
{
  int sum = 0;
  for (const auto* ib : buckets) {
    sum += ib->size();
  }
  return sum;
}

stk::topology::rank_t
get_side_rank(const stk::mesh::BulkData& bulk)
{
  return bulk.mesh_meta_data().side_rank();
}

int
count_faces(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& sel)
{
  return count_ent(bulk.get_buckets(get_side_rank(bulk), sel));
}

int
get_dimension(const stk::mesh::BulkData& bulk)
{
  return int(bulk.mesh_meta_data().spatial_dimension());
}

const stk::mesh::Field<double>&
get_coordinate_field(const stk::mesh::BulkData& bulk)
{
  auto* coord = dynamic_cast<const stk::mesh::Field<double>*>(
    bulk.mesh_meta_data().coordinate_field());
  STK_ThrowRequireMsg(
    coord, "Model coordinates must be a double-valued vector");
  return *coord;
}

void
write_element_block_definition(
  Ioss::Region& region,
  Ioss::DatabaseIO& io,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::ConstPartVector& sides)
{
  for (const auto* side : sides) {
    for (const auto* subset : side->subsets()) {
      auto block = std::make_unique<Ioss::ElementBlock>(
        &io, subset->name(), subset->topology().name(),
        count_faces(bulk, *subset));
      STK_ThrowRequire(block);
      region.add(block.release());
    }
  }
}

void
write_node_block_definition(
  Ioss::Region& region,
  Ioss::DatabaseIO& io,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::ConstPartVector& sides,
  std::string block_name)
{
  const auto& buckets =
    bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(sides));
  const auto nnodes = count_ent(buckets);
  const int dim = get_dimension(bulk);
  auto block = std::make_unique<Ioss::NodeBlock>(&io, block_name, nnodes, dim);
  STK_ThrowRequire(block);
  region.add(block.release());
}

void
write_node_ids(
  Ioss::NodeBlock& block,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::ConstPartVector& parts)
{
  const auto& buckets =
    bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(parts));

  std::vector<int64_t> node_ids;
  node_ids.reserve(count_ent(buckets));

  for (const auto* ib : buckets) {
    for (const auto& node : *ib) {
      node_ids.push_back(bulk.identifier(node));
    }
  }
  block.put_field_data("ids", node_ids);
}

void
write_coordinate_list(
  Ioss::NodeBlock& block,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::ConstPartVector& parts)
{
  const auto& coord_field = get_coordinate_field(bulk);
  const auto& buckets =
    bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectUnion(parts));
  const int dim = get_dimension(bulk);
  std::vector<double> coords;
  coords.reserve(count_ent(buckets) * dim);
  for (const auto* ib : buckets) {
    for (const auto& node : *ib) {
      const auto* xnode = stk::mesh::field_data(coord_field, node);
      for (int k = 0; k < dim; ++k) {
        coords.push_back(xnode[k]);
      }
    }
  }
  block.put_field_data("mesh_model_coordinates", coords);
}

void
write_element_connectivity(
  Ioss::Region& region,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::ConstPartVector& parts)
{
  for (const auto* part : parts) {
    for (const auto* subset : part->subsets()) {
      const auto buckets = bulk.get_buckets(get_side_rank(bulk), *part);

      std::vector<int64_t> connectivity;
      connectivity.reserve([&]() {
        int count = 0;
        for (const auto* ib : buckets) {
          for (const auto& face : *ib) {
            count += bulk.num_nodes(face);
          }
        }
        return count;
      }());

      std::vector<int64_t> ids;
      ids.reserve(count_faces(bulk, *subset));
      for (const auto* ib : buckets) {
        for (const auto& face : *ib) {
          ids.push_back(bulk.identifier(face));
          const auto nodes_per_face = bulk.num_nodes(face);
          const auto nodes = bulk.begin_nodes(face);
          for (unsigned k = 0; k < nodes_per_face; ++k) {
            connectivity.push_back(bulk.identifier(nodes[k]));
          }
        }
      }
      auto block = region.get_element_block(subset->name());
      STK_ThrowRequire(block);
      block->put_field_data("ids", ids);
      block->put_field_data("connectivity", connectivity);
    }
  }
}

template <typename T>
void
put_data_on_node_block(
  Ioss::NodeBlock& block,
  const stk::mesh::BulkData& bulk,
  const std::vector<int64_t>& ids,
  const stk::mesh::FieldBase& field)
{
  STK_ThrowRequire(field.type_is<T>());
  const int max_size = field.max_size();
  std::vector<T> flat_array(ids.size() * max_size, 0);
  for (decltype(ids.size()) k = 0; k < ids.size(); ++k) {
    const auto node = bulk.get_entity(stk::topology::NODE_RANK, ids[k]);
    const T* field_data = static_cast<T*>(stk::mesh::field_data(field, node));
    if (field_data) {
      STK_ThrowRequire(
        stk::mesh::field_scalars_per_entity(field, node) ==
        static_cast<unsigned>(max_size));
      for (int j = 0; j < max_size; ++j) {
        flat_array[k * max_size + j] = field_data[j];
      }
    }
  }
  block.put_field_data(field.name(), flat_array);
}

template <typename... Args>
void
put_data_on_node_block(Args&&... args)
{
  // TODO more than double
  put_data_on_node_block<double>(std::forward<Args>(args)...);
}

} // namespace

SideWriter::SideWriter(
  const stk::mesh::BulkData& bulk,
  std::vector<const stk::mesh::Part*> sides,
  std::vector<const stk::mesh::FieldBase*> fields,
  std::string fname)
  : bulk_(bulk)
{
  Ioss::Init::Initializer init_db;

  Ioss::PropertyManager prop;
  prop.add(Ioss::Property{"INTEGER_SIZE_API", 8});
  prop.add(Ioss::Property{"INTEGER_SIZE_DB", 8});

  auto database = Ioss::IOFactory::create(
    "exodus", fname, Ioss::WRITE_RESULTS, bulk.parallel(), prop);
  STK_ThrowRequire(database != nullptr && database->ok(true));
  output_ = std::make_unique<Ioss::Region>(database, "SideOutput");

  const std::string node_block_name("side_nodes");
  output_->begin_mode(Ioss::STATE_DEFINE_MODEL);
  {
    write_node_block_definition(
      *output_, *database, bulk, sides, node_block_name);
    write_element_block_definition(*output_, *database, bulk, sides);
  }
  output_->end_mode(Ioss::STATE_DEFINE_MODEL);

  output_->begin_mode(Ioss::STATE_MODEL);
  {
    auto& node_block = *output_->get_node_block(node_block_name);
    write_node_ids(node_block, bulk, sides);
    write_coordinate_list(node_block, bulk, sides);
    write_element_connectivity(*output_, bulk, sides);
  }
  output_->end_mode(Ioss::STATE_MODEL);

  output_->begin_mode(Ioss::STATE_DEFINE_TRANSIENT);
  {
    add_fields(fields);
  }
  output_->end_mode(Ioss::STATE_DEFINE_TRANSIENT);
}

void
SideWriter::write_database_data(double time)
{
  output_->begin_mode(Ioss::STATE_TRANSIENT);
  {
    auto current_output_step = output_->add_state(time);
    output_->begin_state(current_output_step);
    {
      for (auto* block : output_->get_node_blocks()) {
        STK_ThrowRequire(block);
        std::vector<int64_t> ids;
        block->get_field_data("ids", ids);
        for (const auto* field : fields_) {
          put_data_on_node_block(*block, bulk_, ids, *field);
        }
      }
    }
    output_->end_state(current_output_step);
  }
  output_->end_mode(Ioss::STATE_TRANSIENT);
}

void
SideWriter::add_fields(std::vector<const stk::mesh::FieldBase*> fields)
{
  for (const auto* field : fields) {
    fields_.insert(field);
  }

  for (auto* block : output_->get_node_blocks()) {
    for (const auto* field : fields_) {
      STK_ThrowRequireMsg(
        field->type_is<double>(), "only double fields supported");
      const size_t nb_size = block->get_property("entity_count").get_int();
      switch (field->max_size()) {
      case 1: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "scalar", Ioss::Field::TRANSIENT,
          nb_size);
        block->field_add(ioss_field);
        break;
      }
      case 2: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "vector_2d",
          Ioss::Field::TRANSIENT, nb_size);
        block->field_add(ioss_field);
        break;
      }
      case 3: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "vector_3d",
          Ioss::Field::TRANSIENT, nb_size);
        block->field_add(ioss_field);
        break;
      }
      case 4: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "full_tensor_22",
          Ioss::Field::TRANSIENT, nb_size);
        block->field_add(ioss_field);
        break;
      }
      case 6: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "sym_tensor_33",
          Ioss::Field::TRANSIENT, nb_size);
        block->field_add(ioss_field);
        break;
      }
      case 9: {
        Ioss::Field ioss_field(
          field->name(), Ioss::Field::DOUBLE, "full_tensor_36",
          Ioss::Field::TRANSIENT, nb_size);
        block->field_add(ioss_field);
        break;
      }
      default:
        throw std::runtime_error(
          "Field type not supported for sideset_writers: " + field->name());
      }
    }
  }
}

void
SideWriterContainer::load(const YAML::Node& node)
{
  const YAML::Node y_writers = node["sideset_writers"];
  if (y_writers) {
    for (size_t i = 0; i < y_writers.size(); ++i) {
      const YAML::Node w_node = y_writers[i];
      std::string name = w_node["name"].as<std::string>();

      outputFileNames_.push_back(
        w_node["output_data_base_name"].as<std::string>());

      outputFrequency_.push_back(w_node["output_frequency"].as<int>());

      const YAML::Node& fromTargets = w_node["target_name"];
      std::vector<std::string> tempPartList;
      if (fromTargets.Type() == YAML::NodeType::Scalar) {
        tempPartList.resize(1);
        tempPartList[0] = fromTargets.as<std::string>();
      } else {
        tempPartList.resize(fromTargets.size());
        for (size_t i = 0; i < fromTargets.size(); ++i) {
          tempPartList[i] = fromTargets[i].as<std::string>();
        }
      }
      sideNames_.push_back(tempPartList);
      const YAML::Node& fieldNames = w_node["output_variables"];
      std::vector<std::string> tempFieldNames;
      if (fieldNames.Type() == YAML::NodeType::Scalar) {
        tempFieldNames.resize(1);
        tempFieldNames[0] = fieldNames.as<std::string>();
      } else {
        tempFieldNames.resize(fieldNames.size());
        for (size_t i = 0; i < fieldNames.size(); ++i) {
          tempFieldNames[i] = fieldNames[i].as<std::string>();
        }
      }
      fieldNames_.push_back(tempFieldNames);
    }
  }
}

void
SideWriterContainer::construct_writers(const stk::mesh::BulkData& bulk)
{
  const auto& meta = bulk.mesh_meta_data();
  for (int i = 0; i < number_of_writers(); i++) {
    // construct part lists
    std::vector<const stk::mesh::Part*> sides;
    for (auto name : sideNames_[i])
      sides.push_back(meta.get_part(name));

    std::vector<const stk::mesh::FieldBase*> fields;
    for (auto name : fieldNames_[i])
      fields.push_back(meta.get_field(stk::topology::NODE_RANK, name));

    sideWriters_.push_back(
      SideWriter(bulk, sides, fields, outputFileNames_[i]));
  }
}

void
SideWriterContainer::write_sides(const int stepCount, const double time)
{
  for (int i = 0; i < number_of_writers(); i++) {
    if (stepCount % outputFrequency_[i] == 0)
      sideWriters_[i].write_database_data(time);
  }
}

} // namespace nalu
} // namespace sierra
