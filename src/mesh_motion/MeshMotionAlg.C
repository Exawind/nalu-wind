
#include "mesh_motion/MeshMotionAlg.h"

#include "mesh_motion/FrameInertial.h"
#include "mesh_motion/FrameNonInertial.h"

#include <cassert>

namespace sierra{
namespace nalu{

MeshMotionAlg::MeshMotionAlg(
Realm& realm,
const YAML::Node& node
) : realm_(realm),
    meta_(*(realm.metaData_)),
    bulk_(*(realm.bulkData_))
{
  load(node);
  post_load();
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

void MeshMotionAlg::post_load()
{
  // compute centroid for all parts as requested in input file
  compute_set_centroid();
}

void MeshMotionAlg::initialize( const double time )
{
  for (size_t i=0; i < frameVec_.size(); i++)
  {
    // set reference frame if they exist
    if( refFrameMap_.find(i) != refFrameMap_.end() )
    {
      int ref_ind = refFrameMap_[i];
      MotionBase::transMatType ref_frame = frameVec_[ref_ind]->get_inertial_frame();
      frameVec_[i]->set_ref_frame(ref_frame);
    }

    frameVec_[i]->setup();

    // update coordinates only if frame is inertial or time > 0.0
    if( (frameVec_[i]->is_inertial()) || (time > 0.0) )
      frameVec_[i]->update_coordinates_velocity(time);
  }
}

void MeshMotionAlg::execute(const double time)
{
  for (size_t i=0; i < frameVec_.size(); i++)
    if( !frameVec_[i]->is_inertial() )
      frameVec_[i]->update_coordinates_velocity(time);
}

void MeshMotionAlg::compute_set_centroid()
{
  std::vector<std::string> partsForCentroid;

  // collect all parts associated with frames instructed to compute their own centroids
  for (size_t i=0; i < frameVec_.size(); i++)
  {
    if( frameVec_[i]->compute_centroid() )
    {
      std::vector<std::string> frameParts = frameVec_[i]->get_part_names();
      partsForCentroid.insert( partsForCentroid.end(), frameParts.begin(), frameParts.end() );
    }
  }

  /// A 3x1 vector that defines the centroid of a collection of parts
  std::vector<double> computedCentroid(MotionBase::threeDVecSize,0.0);

  // compute centroid only if any part was instructed to do so
  if(partsForCentroid.size() > 0)
    realm_.compute_centroid_on_parts( partsForCentroid, computedCentroid );

  // set the centroid for relevant frames
  for (size_t i=0; i < frameVec_.size(); i++)
    if( frameVec_[i]->compute_centroid() )
      frameVec_[i]->set_computed_centroid(computedCentroid);
}

} // nalu
} // sierra
