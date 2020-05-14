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
) : FrameBase(bulk,node,true)
  {
  }

  virtual ~FrameReference()
  {
  }

  void update_coordinates_velocity(const double time);

  const MotionBase::TransMatType& get_reference_frame() const
  {
    return referenceFrame_;
  }

private:
    FrameReference() = delete;
    FrameReference(const FrameReference&) = delete;

    void compute_transformation(const double);

    /** Inertial frame
     *
     * A 4x4 matrix that defines the composite reference frame
     * It is initialized to an identity matrix
     */
    MotionBase::TransMatType referenceFrame_ = MotionBase::identityMat_;
};

} // nalu
} // sierra

#endif /* FRAMEREFERENCE_H */
