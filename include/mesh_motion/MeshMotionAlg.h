#ifndef MESHMOTIONALG_H
#define MESHMOTIONALG_H

#include "FrameMoving.h"

namespace sierra {
namespace nalu {

class MeshMotionAlg
{
public:
  MeshMotionAlg(stk::mesh::BulkData& bulk, const YAML::Node&);

  ~MeshMotionAlg() {}

  void initialize(const double);

  void execute(const double);

  void post_compute_geometry();

  stk::mesh::PartVector get_partvec();

  bool onlyInitialDisplacement_ = true;
  bool is_deforming() { return isDeforming_; }

private:
  MeshMotionAlg() = delete;
  MeshMotionAlg(const MeshMotionAlg&) = delete;

  void load(stk::mesh::BulkData&, const YAML::Node&);

  void set_deformation_flag();

  void compute_set_centroid();

  /** Motion frame vector
   *
   *  Vector of type of frame of corresponding motion
   *  Size is the number of motion groups in input file
   */
  std::vector<std::shared_ptr<FrameBase>> frameVec_;

  /** Reference frame map
   *
   *  Vector of moving frames
   *  Size is the number of groups under mesh_motion in input file
   */
  std::vector<std::shared_ptr<FrameMoving>> movingFrameVec_;

  bool isDeforming_ = false; // flag to denote if mesh deformation exists

  //! flag to guard against multiple invocations of initialize()
  bool isInit_ = false;
};

} // namespace nalu
} // namespace sierra

#endif /* MESHMOTIONALG_H */
