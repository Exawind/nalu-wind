
#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameInertial.h"
#include "mesh_motion/FrameNonInertial.h"

#include <cassert>

namespace sierra{
namespace nalu{

MeshMotionAlg::MeshMotionAlg(
stk::mesh::MetaData& meta,
stk::mesh::BulkData& bulk,
const YAML::Node& node
) : meta_(meta),
    bulk_(bulk)
{
  if( meta_.spatial_dimension() != 3 )
    throw std::runtime_error("MeshMotion: Mesh motion is set up for only 3D meshes");

  load(node);
}

void MeshMotionAlg::load(const YAML::Node& node)
{
  // get motion information for entire mesh
  const auto& minfo = node["motion_group"];
  const int num_groups = minfo.size();
  frameVec_.resize(num_groups);

  // temporary vector to store frame names
  std::vector<std::string> frameNames(num_groups);

  for (int i=0; i < num_groups; i++) {

    // extract current motion group info
    const auto& ginfo = minfo[i];

    // get name of motion group
    frameNames[i] = ginfo["name"].as<std::string>();

    // get frame definition of motion group
    std::string frame = ginfo["frame"].as<std::string>();

    if( frame == "inertial" )
      frameVec_[i].reset(new FrameInertial(meta_, bulk_, ginfo));
    else if( frame == "non_inertial" )
      frameVec_[i].reset(new FrameNonInertial(meta_, bulk_, ginfo));
    else
      throw std::runtime_error("MeshMotion: Invalid frame type: " + frame);

    // get the reference frame if it exists
    if(ginfo["reference"])
    {
      std::string refFrameName = ginfo["reference"].as<std::string>();
      auto it = std::find(frameNames.begin(), frameNames.end(), refFrameName);
      refFrameMap_[i] = std::distance(frameNames.begin(), it);
    }
  }
}

void MeshMotionAlg::initialize( const double time )
{
  for (int i=0; i < frameVec_.size(); i++)
  {
    // set reference frame if they exist
    if( refFrameMap_.find(i) != refFrameMap_.end() )
    {
      int ref_ind = refFrameMap_[i];
      MotionBase::transMatType ref_frame = frameVec_[ref_ind]->get_inertial_frame();
      frameVec_[i]->set_ref_frame(ref_frame);
    }

    // update coordinates only if frame is inertial or time > 0.0
    if( (frameVec_[i]->is_inertial()) || (time > 0.0) )
      frameVec_[i]->update_coordinates_velocity(time);
  }
}

void MeshMotionAlg::execute(const double time)
{
  for (int i=0; i < frameVec_.size(); i++)
    if( !frameVec_[i]->is_inertial() )
      frameVec_[i]->update_coordinates_velocity(time);
}

} // nalu
} // sierra
