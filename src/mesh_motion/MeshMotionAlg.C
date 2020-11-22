
#include "mesh_motion/MeshMotionAlg.h"

#include "NaluParsing.h"

#include <cassert>
#include <iostream>

#include "../../include/mesh_motion/FrameMoving.h"

namespace sierra{
namespace nalu{

MeshMotionAlg::MeshMotionAlg(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node)
  : bulk_(bulk)
{
  load(bulk, node);
}

void MeshMotionAlg::load(
  stk::mesh::BulkData& bulk,
  const YAML::Node& node)
{
  // get motion information for entire mesh
  const int num_groups = node.size();
  movingFrameVec_.resize(num_groups);

  for (int i=0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = node[i];

    movingFrameVec_[i].reset(new FrameMoving(bulk, ginfo));
  }
}

void MeshMotionAlg::initialize( const double time )
{
  if(isInit_)
    throw std::runtime_error("MeshMotionAlg::initialize(): Re-initialization of MeshMotionAlg not valid");

  for (size_t i=0; i < movingFrameVec_.size(); i++)
  {
    movingFrameVec_[i]->setup();

    // update coordinates and velocity
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }

  // Manually synchronize fields to hosts
  {
    const auto& meta = bulk_.mesh_meta_data();
    std::vector<std::string> fnames{
      "current_coordinates",
      "mesh_displacement",
      "mesh_velocity",
    };

    for (const auto& ff: fnames) {
      auto* fld = meta.get_field(stk::topology::NODE_RANK, ff);
      fld->modify_on_device();
      fld->sync_to_host();
    }
  }

  isInit_ = true;
}

void MeshMotionAlg::execute(const double time)
{
  for (size_t i=0; i < movingFrameVec_.size(); i++) {
    movingFrameVec_[i]->update_coordinates_velocity(time);
  }

  // Manually synchronize fields to host
  {
    const auto& meta = bulk_.mesh_meta_data();
    std::vector<std::string> fnames{
      "current_coordinates",
      "mesh_displacement",
      "mesh_velocity",
    };

    for (const auto& ff: fnames) {
      auto* fld = meta.get_field(stk::topology::NODE_RANK, ff);
      fld->modify_on_device();
      fld->sync_to_host();
    }
  }
}

void MeshMotionAlg::post_compute_geometry()
{
  for (size_t i=0; i < movingFrameVec_.size(); i++)
    movingFrameVec_[i]->post_compute_geometry();

  // Manually synchronize fields to device
  {
    auto* divMeshVel = bulk_.mesh_meta_data().get_field(
        stk::topology::NODE_RANK, "div_mesh_velocity");
    if (divMeshVel != nullptr) {
      divMeshVel->modify_on_host();
      divMeshVel->sync_to_device();
    }
  }
}

} // nalu
} // sierra
