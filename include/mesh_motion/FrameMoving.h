#ifndef FRAMEMOVING_H
#define FRAMEMOVING_H

#include "FrameBase.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra{
namespace nalu{

class FrameMoving : public FrameBase
{
public:
  FrameMoving(
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : FrameBase(bulk,node)
  {
  }

  virtual ~FrameMoving()
  {
  }

  void update_coordinates_velocity(const double time);

  void post_compute_geometry();

private:
  FrameMoving() = delete;
  FrameMoving(const FrameMoving&) = delete;
};

} // nalu
} // sierra

#endif /* FRAMEMOVING_H */
