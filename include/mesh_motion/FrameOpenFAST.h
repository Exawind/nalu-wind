#ifndef FRAMEOPENFAST_H
#define FRAMEOPENFAST_H

#include "FrameMoving.h"
#include "FSIturbine.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra {
namespace nalu {

class FrameOpenFAST : public FrameMoving
{
public:
  FrameOpenFAST(
    stk::mesh::BulkData& bulk,
    const YAML::Node& node,
    fsiTurbine* fsiturbinedata)
    : FrameMoving(bulk, node), fsiTurbineData_(fsiturbinedata)
  {
  }

  virtual ~FrameOpenFAST() {}

  void update_coordinates_velocity(const double time);

  void post_compute_geometry(){};

private:
  FrameOpenFAST() = delete;
  FrameOpenFAST(const FrameOpenFAST&) = delete;

  fsiTurbine* fsiTurbineData_;
};

} // namespace nalu
} // namespace sierra

#endif /* FRAMEOPENFAST_H */
