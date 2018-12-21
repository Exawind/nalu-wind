#ifndef FRAMENONINERTIAL_H
#define FRAMENONINERTIAL_H

#include "FrameBase.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra{
namespace nalu{

class FrameNonInertial : public FrameBase
{
public:
  FrameNonInertial(
    stk::mesh::MetaData& meta,
    stk::mesh::BulkData& bulk,
    const YAML::Node& node
) : FrameBase(meta,bulk,node,false) {}

  virtual ~FrameNonInertial() {}

  void update_coordinates_velocity(const double time);

private:
    FrameNonInertial() = delete;
    FrameNonInertial(const FrameNonInertial&) = delete;

    /** Compute transformation matrix
     *
     * @return 4x4 matrix representing composite addition of motions
     */
    MotionBase::transMatType compute_transformation(
      const double,
      const double*);
};

} // nalu
} // sierra

#endif /* FRAMENONINERTIAL_H */
