
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
  stk::mesh::BulkData& bulk,
  const YAML::Node& node,
  bool isInertial
) : bulk_(bulk),
    meta_(bulk.mesh_meta_data()),
    isInertial_(isInertial)
{
  load(node);
}

void FrameBase::load(const YAML::Node& node)
{
  // get any part names associated with current motion group
  populate_part_vec(node);

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
    std::string type;
    get_required(motion_def, "type", type);

    // determine type of mesh motion based on user definition in input file
    if (type == "pulsating_sphere")
      meshMotionVec_[i].reset(new MotionPulsatingSphere(meta_,motion_def));
    else if (type == "rotation")
      meshMotionVec_[i].reset(new MotionRotation(motion_def));
    else if (type == "scaling")
      meshMotionVec_[i].reset(new MotionScaling(motion_def));
    else if (type == "translation")
      meshMotionVec_[i].reset(new MotionTranslation(motion_def));
    else
      throw std::runtime_error("FrameBase: Invalid mesh motion type: " + type);

  } // end for loop - i index
}

void FrameBase::populate_part_vec(const YAML::Node& node)
{
  // if nor parts specified and frame is inertial, return
  if (!node["mesh_parts"] && isInertial_)
    return;

  // declare temporary part name vectors
  std::vector<std::string> partNamesVec;
  std::vector<std::string> partNamesVecBc;

  // populate volume parts
  const auto& fparts = node["mesh_parts"];

  if (fparts.Type() == YAML::NodeType::Scalar)
    partNamesVec.push_back(fparts.as<std::string>());
  else
    partNamesVec = fparts.as<std::vector<std::string>>();

  assert (partNamesVec.size() > 0);

  // store all parts associated with current motion frame
  int numParts = partNamesVec.size();
  partVec_.resize(numParts);

  for (int i=0; i < numParts; i++) {
    stk::mesh::Part* part = meta_.get_part(partNamesVec[i]);
    if (nullptr == part)
      throw std::runtime_error(
        "FrameBase: Invalid part name encountered: " + partNamesVec[i]);
    else
      partVec_[i] = part;
  }

  // populate bc parts if any defined
  if (!node["mesh_parts_bc"])
    return;

  const auto& fpartsBc = node["mesh_parts_bc"];
  if (fpartsBc.Type() == YAML::NodeType::Scalar)
    partNamesVecBc.push_back(fparts.as<std::string>());
  else
    partNamesVecBc = fparts.as<std::vector<std::string>>();

  // store all Bc parts associated with current motion frame
  numParts = partNamesVecBc.size();
  partVecBc_.resize(numParts);

  for (int i=0; i < numParts; i++) {
    stk::mesh::Part* part = meta_.get_part(partNamesVecBc[i]);
    if (nullptr == part)
      throw std::runtime_error(
        "FrameBase: Invalid part name encountered: " + partNamesVecBc[i]);
    else
      partVecBc_[i] = part;
  }
}

void FrameBase::setup()
{
  // compute and set centroid if requested
  if(computeCentroid_) {
    std::vector<double> computedCentroid(3,0.0);
    compute_centroid_on_parts( computedCentroid );
    set_computed_centroid( computedCentroid );
  }
}

void FrameBase::compute_centroid_on_parts(
  std::vector<double>& centroid)
{
  // set min/max
  const int nDim = meta_.spatial_dimension();
  ThrowRequire(nDim <= 3);

  const double largeNumber = 1.0e16;
  double minCoord[3] = {largeNumber, largeNumber, largeNumber};
  double maxCoord[3] = {-largeNumber, -largeNumber, -largeNumber};

  // model coords are fine in this case
  VectorFieldType *modelCoords = meta_.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // select all nodes
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_);

  // select all locally owned nodes for bounding box
  stk::mesh::BucketVector const& bkts = bulk_.get_buckets( stk::topology::NODE_RANK, sel );
  for ( stk::mesh::BucketVector::const_iterator ib = bkts.begin(); ib != bkts.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * mCoord = stk::mesh::field_data(*modelCoords, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      minCoord[0] = std::min(minCoord[0], mCoord[k*nDim+0]);
      maxCoord[0] = std::max(maxCoord[0], mCoord[k*nDim+0]);
      minCoord[1] = std::min(minCoord[1], mCoord[k*nDim+1]);
      maxCoord[1] = std::max(maxCoord[1], mCoord[k*nDim+1]);
      if (nDim == 3) {
        minCoord[2] = std::min(minCoord[2], mCoord[k*nDim+2]);
        maxCoord[2] = std::max(maxCoord[2], mCoord[k*nDim+2]);
      }
    }
  }

  // parallel reduction on min/max
  double g_minCoord[3] = {};
  double g_maxCoord[3] = {};
  stk::ParallelMachine comm = NaluEnv::self().parallel_comm();
  stk::all_reduce_min(comm, minCoord, g_minCoord, 3);
  stk::all_reduce_max(comm, maxCoord, g_maxCoord, 3);

  // ensure the centroid is size number of dimensions
  for ( int j = 0; j < nDim; ++j )
    centroid[j] = 0.5*(g_maxCoord[j] + g_minCoord[j]);
}

} // nalu
} // sierra
