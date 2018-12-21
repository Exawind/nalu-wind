
#include "mesh_motion/FrameBase.h"

#include "mesh_motion/MotionPulsatingSphere.h"
#include "mesh_motion/MotionRotation.h"
#include "mesh_motion/MotionScaling.h"
#include "mesh_motion/MotionTranslation.h"

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
  if (partNamesVec_.size() == 0)
    return;

  VectorFieldType& coordinates = meta_.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType& current_coordinates = meta_.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType& mesh_displacement = meta_.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");
  VectorFieldType& mesh_velocity = meta_.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // store all parts associated with current motion frame
  for (auto pName: partNamesVec_) {
    stk::mesh::Part* part = meta_.get_part(pName);
    if (nullptr == part)
      throw std::runtime_error(
        "MeshMotion: Invalid part name encountered: " + pName);
    else
      partVec_.push_back(part);
  } // end for loop - partNamesVec_

  for (auto* p: partVec_) {
    stk::mesh::put_field(coordinates, *p);
    stk::mesh::put_field(current_coordinates, *p);
    stk::mesh::put_field(mesh_displacement, *p);
    stk::mesh::put_field(mesh_velocity, *p);
  } // end for loop - partVec_
}

} // nalu
} // sierra
