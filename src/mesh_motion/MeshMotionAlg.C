#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameMoving.h"

#include "NaluParsing.h"

#include <cassert>
#include <iostream>

namespace sierra {
namespace nalu {

MeshMotionAlg::MeshMotionAlg(stk::mesh::BulkData& bulk, const YAML::Node& node)
{
  load(bulk, node);

  set_deformation_flag();
}

void
MeshMotionAlg::load(stk::mesh::BulkData& bulk, const YAML::Node& node)
{
  // get motion information for entire mesh
  const int num_groups = node.size();
  movingFrameVec_.resize(num_groups);

  for (int i = 0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = node[i];

    movingFrameVec_[i].reset(new FrameMoving(bulk, ginfo));
  }
}

void
MeshMotionAlg::set_deformation_flag()
{
  for (size_t i = 0; i < movingFrameVec_.size(); i++)
    if (movingFrameVec_[i]->is_deforming())
      isDeforming_ = true;
}

void
MeshMotionAlg::initialize(const double time)
{
  if (isInit_)
    throw std::runtime_error(
      "MeshMotionAlg::initialize(): Re-initialization "
      "of MeshMotionAlg not valid");

  for (size_t i = 0; i < movingFrameVec_.size(); i++) {
    movingFrameVec_[i]->setup();

    // update coordinates and velocity
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }

  isInit_ = true;
}

void
MeshMotionAlg::restart_reinit(const double time)
{
  if (isInit_) {
    isInit_ = false;
    initialize(time);
  } else {
    throw std::runtime_error(
      "MeshMotionAlg::restart_reinit(): Re-initialization of MeshMotionAlg for "
      "restart should be called after initialize");
  }
}

void
MeshMotionAlg::execute(const double time)
{
  for (size_t i = 0; i < movingFrameVec_.size(); i++) {
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }
}

void
MeshMotionAlg::post_compute_geometry()
{
  for (size_t i = 0; i < movingFrameVec_.size(); i++)
    movingFrameVec_[i]->post_compute_geometry();
}

stk::mesh::PartVector
MeshMotionAlg::get_partvec()
{
  stk::mesh::PartVector fpartVec;
  for (size_t i = 0; i < movingFrameVec_.size(); i++) {
    stk::mesh::PartVector fPartVec = movingFrameVec_[i]->get_partvec();
    for (auto p : fPartVec)
      fpartVec.push_back(p);
  }
  return fpartVec;
}

} // namespace nalu
} // namespace sierra
