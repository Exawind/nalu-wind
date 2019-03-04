#ifndef MESHMOTIONALG_H
#define MESHMOTIONALG_H

#include "FrameBase.h"

namespace sierra{
namespace nalu{

class MeshMotionAlg
{
public:
  MeshMotionAlg(
    stk::mesh::BulkData& bulk,
    const YAML::Node&);

  ~MeshMotionAlg() {}

  void initialize(const double);

  void execute(const double);

private:
  MeshMotionAlg() = delete;
  MeshMotionAlg(const MeshMotionAlg&) = delete;

  void load(
    stk::mesh::BulkData&,
    const YAML::Node&);

  void compute_set_centroid();

  /** Motion frame vector
   *
   *  Vector of type of frame of corresponding motion
   *  Size is the number of motion groups in input file
   */
  std::vector<std::shared_ptr<FrameBase>> frameVec_;

  /** Reference frame map
   *
   *  Map between frame indices and corresponding reference frame
   *  Size is the number of motion groups with reference frames
   */
  std::map<int, std::shared_ptr<FrameBase>> refFrameMap_;

  //! flag to guard against multiple invocations of initialize()
  bool isInit_ = false;
};

} // nalu
} // sierra

#endif /* MESHMOTIONALG_H */
