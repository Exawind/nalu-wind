
#include "mesh_motion/FrameMoving.h"

#include "FieldTypeDef.h"

// stk_mesh/base/fem
#include "stk_mesh/base/GetNgpField.hpp"

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

  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "coordinates"));
  stk::mesh::NgpField<double> currCoords = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "current_coordinates"));
  stk::mesh::NgpField<double> displacement = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "mesh_displacement"));
  stk::mesh::NgpField<double> meshVelocity = stk::mesh::get_updated_ngp_field<double>(
    *meta_.get_field<VectorFieldType>(entityRank, "mesh_velocity"));

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_) &
      (meta_.locally_owned_part() | meta_.globally_shared_part());
  const auto& buckets = stk::mesh::get_bucket_ids(bulk_, entityRank, sel);

  // always reset velocity field
//  stk::mesh::field_fill(0.0, meshVelocity, sel);

  // memory requirements for NGP loop
  const int bytes_per_team = 0;
  const int bytes_per_thread = 0;
  auto team_exec = get_device_team_policy(buckets.size(), bytes_per_team, bytes_per_thread);

  // NGP for loop
  Kokkos::parallel_for(
    team_exec, KOKKOS_LAMBDA(const DeviceTeamHandleType& team) {
    auto bktId = buckets.device_get(team.league_rank());
    auto& b = ngpMesh.get_bucket(entityRank, bktId);

    const size_t bktLen = b.size();
    Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, bktLen),
      [&](const size_t& bktIndex) {
        auto node = b[bktIndex];
        const auto nInd = ngpMesh.fast_mesh_index(node);

        // temporary current and model coords for a generic 2D and 3D implementation
        double mX[3] = {0.0,0.0,0.0};
        double cX[3] = {0.0,0.0,0.0};

        // copy over model coordinates and reset velocity
        for (int d = 0; d < nDim; ++d) {
          mX[d] = modelCoords.get(nInd,d);
          meshVelocity.get(nInd,d) = 0.0;
        }

        // all frame motions are based off of the reference frame
        NgpMotion::TransMatType comp_trans_mat = NgpMotion::identity_mat();
        for (size_t i=0; i < numKernels; ++i) {
          NgpMotion* kernel = ngpKernels(i);
          // build and get transformation matrix
          kernel->build_transformation(time,mX);
          // composite addition of motions in current group
          comp_trans_mat = kernel->add_motion(kernel->get_trans_mat(),comp_trans_mat);
        }

        // perform matrix multiplication between transformation matrix
        // and old coordinates to obtain current coordinates
        for (int d = 0; d < nDim; ++d) {
          currCoords.get(nInd,d) = comp_trans_mat[d][0]*mX[0]
                                  +comp_trans_mat[d][1]*mX[1]
                                  +comp_trans_mat[d][2]*mX[2]
                                  +comp_trans_mat[d][3];

          displacement.get(nInd,d) = currCoords.get(nInd,d) - modelCoords.get(nInd,d);
        } // end for loop - d index

        // copy over current coordinates
        for (int d = 0; d < nDim; ++d)
          cX[d] = currCoords.get(nInd,d);

        // compute velocity vector on current node resulting from all
        // motions in current motion frame
        for (size_t i=0; i < numKernels; ++i) {
          NgpMotion* kernel = ngpKernels(i);
          // build and get transformation matrix
          NgpMotion::ThreeDVecType mm_vel = kernel->compute_velocity(time,comp_trans_mat,mX,cX);

          for (int d = 0; d < nDim; ++d)
            meshVelocity.get(nInd,d) += mm_vel[d];
        } // end for loop - mm
    });
  }); // end for loop - bkts
}

void FrameMoving::post_compute_geometry()
{
  // flag denoting if mesh velocity divergence already computed
  bool computedMeshVelDiv = false;

  for (auto& mm: motionKernels_)
    mm->post_compute_geometry(bulk_,partVec_,partVecBc_,computedMeshVelDiv);
}

} // nalu
} // sierra
