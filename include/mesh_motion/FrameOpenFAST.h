#ifndef FRAMEOPENFAST_H
#define FRAMEOPENFAST_H

#include "FrameBase.h"
#include "FSIturbine.h"

#include "yaml-cpp/yaml.h"

#include <cassert>
#include <float.h>

namespace sierra{
namespace nalu{

class FrameOpenFAST : public FrameBase
{
public:
  FrameOpenFAST(
    stk::mesh::BulkData& bulk,
    YAML::Node node,
    fsiTurbine* fsiturbinedata
  ) : FrameBase(bulk,node),
      fsiTurbineData_(fsiturbinedata)
  {
  }

  virtual ~FrameOpenFAST()
  {
  }

  void update_coordinates_velocity(const double time);

  void post_compute_geometry();

private:
    FrameOpenFAST() = delete;
    FrameOpenFAST(const FrameOpenFAST&) = delete;

    fsiTurbine* fsiTurbineData_;
};

} // nalu
} // sierra

#endif /* FRAMEOPENFAST_H */
