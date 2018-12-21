#ifndef MESHMOTIONALG_H
#define MESHMOTIONALG_H

#include "FrameBase.h"

namespace sierra{
namespace nalu{

class MeshMotionAlg
{
public:
  MeshMotionAlg(
    stk::mesh::MetaData&,
    stk::mesh::BulkData&,
    const YAML::Node&);

  ~MeshMotionAlg() {}

  void initialize(const double);

  void execute(const double);

private:
  MeshMotionAlg() = delete;
  MeshMotionAlg(const MeshMotionAlg&) = delete;

  void load(const YAML::Node&);

  //! Reference to the STK Mesh MetaData object
  stk::mesh::MetaData& meta_;

  //! Reference to the STK Mesh BulkData object
  stk::mesh::BulkData& bulk_;

  /** Motion frame vector
   *
   *  Vector of type of frame of corresponding motion
   *  Size is the number of motion groups in input file
   */
  std::vector<std::unique_ptr<FrameBase>> frameVec_;

  /** Reference frame map
   *
   *  Map between frame indices and corresponding reference frame indices
   *  Size is the number of motion groups with reference frames
   */
  std::map<int, int> refFrameMap_;
};

} // nalu
} // sierra

#endif /* MESHMOTIONALG_H */
