
#include "mesh_motion/FrameBase.h"

#include "mesh_motion/MotionPulsatingSphere.h"
#include "mesh_motion/MotionRotation.h"
#include "mesh_motion/MotionScaling.h"
#include "mesh_motion/MotionTranslation.h"

#include <NaluParsing.h>

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra{
namespace nalu{

FrameBase::FrameBase(
  stk::mesh::MetaData& meta,
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  bool isInertial
) : meta_(meta),
    bulk_(bulk),
    isInertial_(isInertial)
{
  load(node);
}

void FrameBase::load(const YAML::Node& node)
{
  // get any part names associated with current motion group
  if (node["mesh_parts"])
  {
    const auto& fparts = node["mesh_parts"];
    partNamesVec_ = fparts.as<std::vector<std::string>>();
  }

  // check if centroid needs to be computed
  get_if_present(node, "compute_centroid", computeCentroid_, computeCentroid_);

  // extract the motions in the current group
  const auto& motions = node["motion"];

  const int num_motions = motions.size();
  meshMotionVec_.resize(num_motions);

  // create the classes associated with every motion in current group
  for (int i=0; i < num_motions; i++) {

    // get the motion definition for i-th transformation
    const auto& motion_def = motions[i];

    // motion type should always be defined by the user
    std::string type = motion_def["type"].as<std::string>();

    // determine type of mesh motion based on user definition in input file
    if (type == "pulsating_sphere")
      meshMotionVec_[i].reset(new MotionPulsatingSphere(motion_def));
    else if (type == "rotation")
      meshMotionVec_[i].reset(new MotionRotation(motion_def));
    else if (type == "scaling")
      meshMotionVec_[i].reset(new MotionScaling(motion_def));
    else if (type == "translation")
      meshMotionVec_[i].reset(new MotionTranslation(motion_def));
    else
      throw std::runtime_error("MeshMotion: Invalid mesh motion type: " + type);

  } // end for loop - i index
}

} // nalu
} // sierra
