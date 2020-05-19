
#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameInertial.h"
#include "mesh_motion/FrameNonInertial.h"

#include "NaluParsing.h"

#include <cassert>
#include <iostream>

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
  frameVec_.resize(num_groups);

  // temporary vector to store frame names
  std::vector<std::string> frameNames(num_groups);

  for (int i=0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = node[i];

    // get name of motion group
    frameNames[i] = ginfo["name"].as<std::string>();

    // get frame definition of motion group
    std::string frame;
    get_required(ginfo, "frame", frame);

    if( frame == "inertial" )
      frameVec_[i].reset(new FrameInertial(bulk, ginfo));
    else if( frame == "non_inertial" )
      frameVec_[i].reset(new FrameNonInertial(bulk, ginfo));
    else
      throw std::runtime_error("MeshMotion: Invalid frame type: " + frame);

    // get the reference frame index if it exists
    if(ginfo["reference"])
    {
      std::string refFrameName = ginfo["reference"].as<std::string>();

      auto it = std::find(frameNames.begin(), frameNames.end(), refFrameName);

      if( it ==  frameNames.end() )
        throw std::runtime_error("MeshMotion: Invalid reference frame: " + refFrameName);

      refFrameMap_[i] = frameVec_[std::distance(frameNames.begin(), it)];
    }
  }
}

void MeshMotionAlg::initialize( const double time )
{
  if(isInit_)
    throw std::runtime_error("MeshMotionAlg::initialize(): Re-initialization of MeshMotionAlg not valid");

  for (size_t i=0; i < frameVec_.size(); i++)
  {
    frameVec_[i]->setup();

    // set reference frame if they exist
    if( refFrameMap_.find(i) != refFrameMap_.end() )
    {
      MotionBase::TransMatType ref_frame = refFrameMap_[i]->get_inertial_frame();
      frameVec_[i]->set_ref_frame(ref_frame);
    }

    // update coordinates and velocity
    frameVec_[i]->update_coordinates_velocity(time);
  }

  // TODO: NGP Transition
  // Manually synchronize fields to device
  {
    const auto& meta = bulk_.mesh_meta_data();
    std::vector<std::string> fnames{
      "current_coordinates",
      "mesh_displacement",
      "mesh_velocity",
    };

    for (const auto& ff: fnames) {
      auto* fld = meta.get_field(stk::topology::NODE_RANK, ff);
      fld->modify_on_host();
      fld->sync_to_device();
    }
  }

  isInit_ = true;
}

void MeshMotionAlg::execute(const double time)
{
  for (size_t i=0; i < frameVec_.size(); i++) {

    if( !frameVec_[i]->is_inertial() )
      frameVec_[i]->update_coordinates_velocity(time);
  }

  // TODO: NGP Transition
  // Manually synchronize fields to device
  {
    const auto& meta = bulk_.mesh_meta_data();
    std::vector<std::string> fnames{
      "current_coordinates",
      "mesh_displacement",
      "mesh_velocity",
    };

    for (const auto& ff: fnames) {
      auto* fld = meta.get_field(stk::topology::NODE_RANK, ff);
      fld->modify_on_host();
      fld->sync_to_device();
    }
  }
}

void MeshMotionAlg::post_compute_geometry()
{
  for (size_t i=0; i < frameVec_.size(); i++)
    frameVec_[i]->post_compute_geometry();

  // TODO: NGP Transition
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
