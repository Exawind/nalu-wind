#ifndef FRAMEREFERENCE_H
#define FRAMEREFERENCE_H

#include "FrameBase.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra {
namespace kynema_ugf {

class FrameReference : public FrameBase
{
public:
  FrameReference(stk::mesh::BulkData& bulk, const YAML::Node& node)
    : FrameBase(bulk, node)
  {
  }

  virtual ~FrameReference() {}

  void update_coordinates(const double time);

private:
  FrameReference() = delete;
  FrameReference(const FrameReference&) = delete;
};

} // namespace kynema_ugf
} // namespace sierra

#endif /* FRAMEREFERENCE_H */
