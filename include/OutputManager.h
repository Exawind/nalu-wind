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
  std::vector<OutputInfo> infoVec_;
  std::vector<size_t> indexVec_;

public:
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
};
} // namespace nalu
} // namespace sierra

#endif /* OUTPUTMANAGER_H_ */
