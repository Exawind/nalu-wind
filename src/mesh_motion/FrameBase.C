#include "mesh_motion/FrameBase.h"

#include "FieldTypeDef.h"
#include "mesh_motion/MotionDeformingInteriorKernel.h"
#include "mesh_motion/MotionScalingKernel.h"
#include "mesh_motion/MotionRotationKernel.h"
#include "mesh_motion/MotionTranslationKernel.h"
#include "mesh_motion/MotionWavesKernel.h"
#include "NaluParsing.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpReducers.h"
#include "ngp_utils/NgpTypes.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>
#include "stk_mesh/base/GetNgpMesh.hpp"

namespace sierra{
namespace nalu{

FrameBase::FrameBase(stk::mesh::BulkData& bulk, const YAML::Node& node)
  : bulk_(bulk), meta_(bulk.mesh_meta_data())
{
  load(node);

  // set deformation flag based on motions in the frame
  for (auto& mm: motionKernels_)
    if ( mm->is_deforming() )
      isDeforming_ = true;
}

FrameBase::~FrameBase()
{
  // Release the device pointers if any
  for (auto& kern: motionKernels_) {
    kern->free_on_device();
  }
}

void
FrameBase::load(const YAML::Node& node)
{
  // get any part names associated with current motion group
  populate_part_vec(node);

  // check if centroid needs to be computed
  get_if_present(node, "compute_centroid", computeCentroid_, computeCentroid_);

  if (node["motion"]) {
      // extract the motions in the current group
      const auto& motions = node["motion"];

  const int num_motions = motions.size();
  motionKernels_.resize(num_motions);

  // create the classes associated with every motion in current group
  for (int i=0; i < num_motions; i++) {

    // get the motion definition for i-th transformation
    const auto& motion_def = motions[i];

    // motion type should always be defined by the user
    std::string type;
    get_required(motion_def, "type", type);

    // determine type of mesh motion based on user definition in input file
    if (type == "deforming_interior")
      motionKernels_[i].reset(new MotionDeformingInteriorKernel(meta_, motion_def));
    else if (type == "rotation")
      motionKernels_[i].reset(new MotionRotationKernel(motion_def));
    else if (type == "scaling")
      motionKernels_[i].reset(new MotionScalingKernel(meta_,motion_def));
    else if (type == "translation")
      motionKernels_[i].reset(new MotionTranslationKernel(motion_def));
    else if (type == "waving_boundary")
      motionKernels_[i].reset(new MotionWavesKernel(meta_,motion_def));
    else
      throw std::runtime_error("FrameBase: Invalid mesh motion type: " + type);

      } // end for loop - i index
  }
}

void
FrameBase::populate_part_vec(const YAML::Node& node)
{

  if (!node["mesh_parts"]) {
    throw std::runtime_error("FrameBase: No mesh parts found.");
  }

  // declare temporary part name vectors
  std::vector<std::string> partNamesVec;
  std::vector<std::string> partNamesVecBc;

  // populate volume parts
  const auto& fparts = node["mesh_parts"];
  if (fparts.Type() == YAML::NodeType::Scalar)
    partNamesVec.push_back(fparts.as<std::string>());
  else
    partNamesVec = fparts.as<std::vector<std::string>>();

  // get all mesh parts if all blocks were requested
  if (
    std::find(partNamesVec.begin(), partNamesVec.end(), "all_blocks") !=
    partNamesVec.end()) {
    partNamesVec.clear();
    for (const auto* part : meta_.get_mesh_parts()) {
      ThrowRequire(part);
      if (part->topology().rank() == stk::topology::ELEMENT_RANK) {
        partNamesVec.push_back(part->name());
      }
    }
  }

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

  std::cerr << " Registering bc parts for mesh motion" << std::endl;

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

void
FrameBase::setup()
{
  // compute and set centroid if requested
  if(computeCentroid_) {
    mm::ThreeDVecType computedCentroid;
    compute_centroid_on_parts( computedCentroid );
    set_computed_centroid( computedCentroid );
  }
}

void
FrameBase::compute_centroid_on_parts(mm::ThreeDVecType& centroid)
{
  // get NGP mesh
  const auto& ngpMesh = stk::mesh::get_updated_ngp_mesh(bulk_);
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords =
    stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "coordinates"));

  // sync fields to device
  modelCoords.sync_to_device();

  // select all nodes in the parts
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_);

  nalu_ngp::MinMaxSumScalar<double> xCoord, yCoord, zCoord;
  nalu_ngp::MinMaxSum<double> xReducer(xCoord), yReducer(yCoord),
    zReducer(zCoord);

  nalu_ngp::run_entity_par_reduce(
    "FrameBase::compute_x_centroid_on_parts", ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(
      const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi,
      nalu_ngp::MinMaxSumScalar<double>& threadVal){
      const double xc = modelCoords.get(mi,0);

      if (xc < threadVal.min_val)
        threadVal.min_val = xc;
      if (xc > threadVal.max_val)
        threadVal.max_val = xc;
    },
    xReducer);
  nalu_ngp::run_entity_par_reduce(
    "FrameBase::compute_y_centroid_on_parts", ngpMesh, stk::topology::NODE_RANK,
    sel,
    KOKKOS_LAMBDA(
      const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi,
      nalu_ngp::MinMaxSumScalar<double>& threadVal){
      const double yc = modelCoords.get(mi,1);

      if (yc < threadVal.min_val)
        threadVal.min_val = yc;
      if (yc > threadVal.max_val)
        threadVal.max_val = yc;
    },
    yReducer);
  nalu_ngp::run_entity_par_reduce(
    "FrameBase::compute_z_centroid_on_parts", ngpMesh, stk::topology::NODE_RANK,
    sel,
    KOKKOS_LAMBDA(
      const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi,
      nalu_ngp::MinMaxSumScalar<double>& threadVal){
      const double zc = modelCoords.get(mi,2);

      if (zc < threadVal.min_val)
        threadVal.min_val = zc;
      if (zc > threadVal.max_val)
        threadVal.max_val = zc;
    },
    zReducer);

  double lXC[2] = {xCoord.min_val, xCoord.max_val};
  double lYC[2] = {yCoord.min_val, yCoord.max_val};
  double lZC[2] = {zCoord.min_val, zCoord.max_val};

  double gXC[2], gYC[2], gZC[2] = {0.0, 0.0};

  stk::all_reduce_min(bulk_.parallel(), &lXC[0], &gXC[0], 1);
  stk::all_reduce_min(bulk_.parallel(), &lYC[0], &gYC[0], 1);
  stk::all_reduce_min(bulk_.parallel(), &lZC[0], &gZC[0], 1);

  stk::all_reduce_max(bulk_.parallel(), &lXC[1], &gXC[1], 1);
  stk::all_reduce_max(bulk_.parallel(), &lYC[1], &gYC[1], 1);
  stk::all_reduce_max(bulk_.parallel(), &lZC[1], &gZC[1], 1);

  // ensure the centroid is size number of dimensions
  centroid[0] = 0.5*(gXC[0] + gXC[1]);
  centroid[1] = 0.5*(gYC[0] + gYC[1]);
  centroid[2] = 0.5*(gZC[0] + gZC[1]);
}

} // namespace nalu
} // namespace sierra
