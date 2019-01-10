
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
  Realm& realm,
  const YAML::Node& node,
  bool isInertial
) : realm_(realm),
    meta_(*(realm.metaData_)),
    bulk_(*(realm.bulkData_)),
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

    if (fparts.Type() == YAML::NodeType::Scalar)
      partNamesVec_.push_back(fparts.as<std::string>());
    else
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

void FrameBase::setup()
{
  // check if any parts have been associated with current frame
  if (partNamesVec_.size() == 0 && isInertial_)
    return;
  else
    assert (partNamesVec_.size() > 0);

  // store all parts associated with current motion frame
  int numParts = partNamesVec_.size();
  partVec_.resize(numParts);

  for (int i=0; i < numParts; i++) {
    stk::mesh::Part* part = meta_.get_part(partNamesVec_[i]);
    if (nullptr == part)
      throw std::runtime_error(
        "FrameBase: Invalid part name encountered: " + partNamesVec_[i]);
    else
      partVec_[i] = part;
  }

  // compute and set centroid if requested
  if(computeCentroid_) {
    std::vector<double> computedCentroid(3,0.0);
    realm_.compute_centroid_on_parts( partNamesVec_, computedCentroid );
    set_computed_centroid( computedCentroid );
  }
}

} // nalu
} // sierra
