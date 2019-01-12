
#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameInertial.h"
#include "mesh_motion/FrameNonInertial.h"

#include <cassert>
#include <iostream>

namespace sierra{
namespace nalu{

MeshMotionAlg::MeshMotionAlg(
Realm& realm,
const YAML::Node& node
) : realm_(realm)
{
  load(node);
}

void MeshMotionAlg::load(const YAML::Node& node)
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
      frameVec_[i].reset(new FrameInertial(realm_, ginfo));
    else if( frame == "non_inertial" )
      frameVec_[i].reset(new FrameNonInertial(realm_, ginfo));
    else
      throw std::runtime_error("MeshMotion: Invalid frame type: " + frame);

    // get the reference frame index if it exists
    if(ginfo["reference"])
    {
      std::string refFrameName = ginfo["reference"].as<std::string>();

      auto it = std::find(frameNames.begin(), frameNames.end(), refFrameName);

      if( it ==  frameNames.end() )
        throw std::runtime_error("MeshMotion: Invalid reference frame: " + refFrameName);

      refFrameMap_[i] = std::distance(frameNames.begin(), it);
    }
  }
}

void MeshMotionAlg::initialize( const double time )
{
  for (size_t i=0; i < frameVec_.size(); i++)
  {
    frameVec_[i]->setup();

    // set reference frame if they exist
    if( refFrameMap_.find(i) != refFrameMap_.end() )
    {
      int ref_ind = refFrameMap_[i];
      MotionBase::transMatType ref_frame = frameVec_[ref_ind]->get_inertial_frame();
      frameVec_[i]->set_ref_frame(ref_frame);
    }

    // update coordinates and velocity
    frameVec_[i]->update_coordinates_velocity(time);
  }
}

void MeshMotionAlg::execute(const double time)
{
  for (size_t i=0; i < frameVec_.size(); i++)
    if( !frameVec_[i]->is_inertial() )
      frameVec_[i]->update_coordinates_velocity(time);
}

} // nalu
} // sierra
