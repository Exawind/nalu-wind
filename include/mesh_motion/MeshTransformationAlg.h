#ifndef MESHTRANSFORMATIONALG_H
#define MESHTRANSFORMATIONALG_H

#include "FrameReference.h"

namespace sierra{
namespace nalu{

class MeshTransformationAlg
{
public:
  MeshTransformationAlg(
    stk::mesh::BulkData& bulk,
    const YAML::Node&);

  ~MeshTransformationAlg()
  {
  }

  void initialize(const double);

  void execute(const double);

  void post_compute_geometry();

private:
  MeshTransformationAlg() = delete;
  MeshTransformationAlg(const MeshTransformationAlg&) = delete;

  void load(
    stk::mesh::BulkData&,
    const YAML::Node&);

  /** Reference frame vector
   *
   *  Vector of reference frames
   *  Size is the number of groups under mesh_transformation in input file
   */
  std::vector<std::shared_ptr<FrameReference>> referenceFrameVec_;

  //! flag to guard against multiple invocations of initialize()
  bool isInit_ = false;
};

} // nalu
} // sierra

#endif /* MESHTRANSFORMATIONALG_H */
