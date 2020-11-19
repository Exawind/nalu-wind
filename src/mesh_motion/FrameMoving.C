
#include "mesh_motion/FrameMoving.h"

#include "FieldTypeDef.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "utils/ComputeVectorDivergence.h"

#include <cassert>

namespace sierra{
namespace nalu{

void FrameMoving::update_coordinates_velocity(const double time)
{
  assert (partVec_.size() > 0);

  // create NGP view of motion kernels
  const size_t numKernels = motionKernels_.size();
  auto ngpKernels = nalu_ngp::create_ngp_view<NgpMotion>(motionKernels_);

  // define mesh entities
  const int nDim = meta_.spatial_dimension();
  const auto& ngpMesh = bulk_.get_updated_ngp_mesh();
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_) &
      (meta_.locally_owned_part() | meta_.globally_shared_part());

  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "coordinates"));
  stk::mesh::NgpField<double> currCoords = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "current_coordinates"));
  stk::mesh::NgpField<double> displacement = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "mesh_displacement"));
  stk::mesh::NgpField<double> meshVelocity = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "mesh_velocity"));

  // always reset velocity field
  nalu_ngp::run_entity_algorithm(
    "FrameMoving_reset_velocity",
    ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {
    for (int d = 0; d < nDim; ++d)
      meshVelocity.get(mi,d) = 0.0;
  });

  // NGP for loop to update coordinates and velocity
  nalu_ngp::run_entity_algorithm(
    "FrameMoving_update_coordinates_velocity",
    ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {

    // temporary current and model coords for a generic 2D and 3D implementation
    NgpMotion::ThreeDVecType mX = {};
    NgpMotion::ThreeDVecType cX = {};

    // copy over model coordinates and reset velocity
    for (int d = 0; d < nDim; ++d)
      mX[d] = modelCoords.get(mi,d);

    // initialize composite transformation matrix
    NgpMotion::TransMatType comp_trans_mat;
    NgpMotion::reset_mat(comp_trans_mat);

    // create composite transformation matrix based off of all motions
    for (size_t i=0; i < numKernels; ++i) {
      NgpMotion* kernel = ngpKernels(i);

      // build and get transformation matrix
      kernel->build_transformation(time,mX);

      // composite addition of motions in current group
      NgpMotion::TransMatType temp_trans_mat = {};
      kernel->add_motion(kernel->get_trans_mat(),comp_trans_mat,temp_trans_mat);
      NgpMotion::copy_mat(comp_trans_mat,temp_trans_mat);
    }

    // perform matrix multiplication between transformation matrix
    // and old coordinates to obtain current coordinates
    for (int d = 0; d < nDim; ++d) {
      currCoords.get(mi,d) = comp_trans_mat[d][0]*mX[0]
                              +comp_trans_mat[d][1]*mX[1]
                              +comp_trans_mat[d][2]*mX[2]
                              +comp_trans_mat[d][3];

      displacement.get(mi,d) = currCoords.get(mi,d) - modelCoords.get(mi,d);
    } // end for loop - d index

    // copy over current coordinates
    for (int d = 0; d < nDim; ++d)
      cX[d] = currCoords.get(mi,d);

    // compute velocity vector on current node resulting from all
    // motions in current motion frame
    for (size_t i=0; i < numKernels; ++i) {
      NgpMotion* kernel = ngpKernels(i);

      // evaluate velocity associated with motion
      NgpMotion::ThreeDVecType mm_vel = {};
      kernel->compute_velocity(time,comp_trans_mat,mX,cX,mm_vel);

      for (int d = 0; d < nDim; ++d)
        meshVelocity.get(mi,d) += mm_vel[d];
    } // end for loop - mm
  }); // end NGP for loop
}

void FrameMoving::post_compute_geometry()
{
  for (auto& mm: motionKernels_) {
    if (!mm->is_deforming())
      continue;

    // compute divergence of mesh velocity
    VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity");
    ScalarFieldType* meshDivVelocity = meta_.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "div_mesh_velocity");
    compute_vector_divergence(bulk_, partVec_, partVecBc_, meshVelocity, meshDivVelocity, true);

    // Mesh velocity divergence is not motion-specific and
    // is computed for the aggregated mesh velocity
    break;
  }
}

} // nalu
} // sierra
