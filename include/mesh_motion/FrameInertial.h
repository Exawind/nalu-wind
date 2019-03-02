#ifndef FRAMEINERTIAL_H
#define FRAMEINERTIAL_H

#include "FrameBase.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra{
namespace nalu{

class FrameInertial : public FrameBase
{
public:
  FrameInertial(
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : FrameBase(bulk,node,true)
  {
  }

  virtual ~FrameInertial()
  {
  }

  void update_coordinates_velocity(const double time);

  const MotionBase::TransMatType& get_inertial_frame() const
  {
    return inertialFrame_;
  }

private:
    FrameInertial() = delete;
    FrameInertial(const FrameInertial&) = delete;

    void compute_transformation(const double);

    /** Inertial frame
     *
     * A 4x4 matrix that defines the composite inertial frame
     * It is initialized to an identity matrix
     */
    MotionBase::TransMatType inertialFrame_ = MotionBase::identityMat_;
};

} // nalu
} // sierra

#endif /* FRAMEINERTIAL_H */
