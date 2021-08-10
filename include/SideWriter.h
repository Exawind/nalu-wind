// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SIDEWRITER_H
#define SIDEWRITER_H

#include <set>
#include <vector>
#include <string>
#include <memory>

namespace Ioss {
class Region;
} // namespace Ioss

namespace YAML {
class Node;
}
namespace stk {
namespace mesh {
class BulkData;
class Selector;
class FieldBase;
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class SideWriter
{
public:
  SideWriter(
    const stk::mesh::BulkData& bulk,
    std::vector<const stk::mesh::Part*> sides,
    std::vector<const stk::mesh::FieldBase*> fields,
    std::string fname);

  void write_database_data(double time);

private:
  void add_fields(std::vector<const stk::mesh::FieldBase*> fields);
  const stk::mesh::BulkData& bulk_;
  std::unique_ptr<Ioss::Region> output_;
  std::set<const stk::mesh::FieldBase*> fields_;
};

class SideWriterContainer
{
public:
  void load(const YAML::Node& node);
  void construct_writers(const stk::mesh::BulkData& bulk);
  void write_sides(const int stepCount, const double time);
  // use outputFileNames since writers have to be constructed
  // so this can be used before the construction
  inline int number_of_writers() { return outputFileNames_.size(); };

private:
  std::vector<SideWriter> sideWriters_;
  std::vector<std::string> outputFileNames_;
  std::vector<int> outputFrequency_;
  std::vector<std::vector<std::string>> sideNames_;
  std::vector<std::vector<std::string>> fieldNames_;
};
} // namespace nalu
} // namespace sierra
#endif
