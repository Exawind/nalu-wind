// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef OUTPUTMANAGER_H_
#define OUTPUTMANAGER_H_

#include <OutputInfo.h>
#include <vector>
#include <string>
#include <stk_mesh/base/Types.hpp> // for EntityRank, etc

namespace stk {
namespace io {
class StkMeshIoBroker;
}
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {
class OutputInfo;

class OutputManager
{
public:
  std::vector<OutputInfo> infoVec_;
  std::vector<size_t> indexVec_;
  // allow for only one catalyst and restart output
  bool hasCatalystOutput_{false};
  int catalystInfoId_{-1};
  bool hasRestartBlock_{false};
  int restartInfoId_{-1};

  OutputManager() {}
  void load(const YAML::Node& node);
  void create_output_mesh(
    stk::io::StkMeshIoBroker* ioBroker, stk::mesh::MetaData* metaData);
  void perform_outputs(
    const int timeStepCount,
    const double currentTime,
    stk::io::StkMeshIoBroker* ioBroker,
    stk::mesh::MetaData* metaData,
    const double wallTimeStart);
  OutputInfo& get_catalyst_output_info() { return infoVec_[catalystInfoId_]; }
  inline bool has_catalyst_output() { return hasCatalystOutput_; }
  OutputInfo& get_restart_output_info() { return infoVec_[restartInfoId_]; }
  inline bool has_restart_output() { return hasRestartBlock_; }
  stk::mesh::EntityRank
  get_entity_rank(const OutputInfo* oInfo, const stk::mesh::MetaData* metaData);
  int serializedIOGroupSize_{0};
};
} // namespace nalu
} // namespace sierra

#endif /* OUTPUTMANAGER_H_ */
