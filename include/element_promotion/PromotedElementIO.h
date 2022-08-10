// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef PromotedElementIO_h
#define PromotedElementIO_h

#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <Ioss_Region.h>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Types.hpp>

#include <element_promotion/HexNElementDescription.h>

#include <stddef.h>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace Ioss {
class DatabaseIO;
class ElementBlock;
class NodeBlock;
class SideBlock;
} // namespace Ioss
namespace sierra {
namespace nalu {
class PromoteElement;
struct ElementDescription;
} // namespace nalu
} // namespace sierra

// field types
typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag> GenericFieldType;
typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;

namespace stk {
namespace mesh {
class BulkData;
class MetaData;
class Part;

typedef std::vector<Part*> PartVector;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class PromotedElementIO
{

public:
  // constructor/destructor
  PromotedElementIO(
    int p,
    const stk::mesh::MetaData& metaData,
    stk::mesh::BulkData& bulkData,
    const stk::mesh::PartVector& baseParts,
    const std::string& fileName,
    const VectorFieldType& coordField);

  virtual ~PromotedElementIO() = default;

  void add_fields(const std::vector<stk::mesh::FieldBase*>& fields);
  std::map<const std::string, const stk::mesh::FieldBase*> get_output_fields()
  {
    return fields_;
  }
  bool has_field(const std::string field_name)
  {
    return (fields_.find(field_name) != fields_.end());
  }
  void write_database_data(double currentTime);

private:
  void
  output_results(const std::vector<const stk::mesh::FieldBase*> fields) const;

  void write_element_connectivity(
    const stk::mesh::PartVector& baseParts,
    const std::vector<stk::mesh::EntityId>& entityIds);

  size_t sub_element_global_id() const;
  void
  write_node_block_definitions(const stk::mesh::PartVector& superElemParts);
  void write_elem_block_definitions(const stk::mesh::PartVector& baseParts);
  void write_coordinate_list(const stk::mesh::PartVector& superElemParts);

  template <typename T>
  void put_data_on_node_block(
    Ioss::NodeBlock& nodeBlock,
    const std::vector<int64_t>& ids,
    const stk::mesh::FieldBase& field,
    const stk::mesh::BucketVector& buckets) const;

  std::string storage_name(const stk::mesh::FieldBase& field) const;

  // meta, bulk and io
  const HexNElementDescription elem_;
  const stk::mesh::MetaData& metaData_;
  const stk::mesh::BulkData& bulkData_;
  const std::string& fileName_;
  const VectorFieldType& coordinates_;
  const unsigned nDim_;
  stk::mesh::PartVector superElemParts_;

  std::map<const std::string, const stk::mesh::FieldBase*> fields_;
  std::map<const stk::mesh::Part*, Ioss::ElementBlock*> elementBlockPointers_;
  std::map<const stk::mesh::Part*, Ioss::SideBlock*> sideBlockPointers_;
  Ioss::NodeBlock* nodeBlock_;

  std::unique_ptr<Ioss::Region> output_;
  Ioss::DatabaseIO* databaseIO;
};

} // namespace nalu
} // namespace sierra

#endif
