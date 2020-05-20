#ifndef FRAMEREFERENCE_H
#define FRAMEREFERENCE_H

#include "FrameBase.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra{
namespace nalu{

class FrameReference : public FrameBase
{
public:
  FrameReference(
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : FrameBase(bulk,node)
  {
  }

  virtual ~FrameReference()
  {
  }

  void update_coordinates(const double time);

private:
    FrameReference() = delete;
    FrameReference(const FrameReference&) = delete;
};

} // nalu
} // sierra

#endif /* FRAMEREFERENCE_H */
