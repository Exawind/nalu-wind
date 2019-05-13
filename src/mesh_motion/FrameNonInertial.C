
#include "mesh_motion/FrameNonInertial.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cassert>

namespace sierra{
namespace nalu{

void FrameNonInertial::update_coordinates_velocity(const double time)
{
  assert (partVec_.size() > 0);

  const int nDim = meta_.spatial_dimension();

  VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currCoords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* displacement = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");
  VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_) &
      (meta_.locally_owned_part() | meta_.globally_shared_part());
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

  // always reset velocity field
  stk::mesh::field_fill(0.0, *meshVelocity, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oldxyz = stk::mesh::field_data(*modelCoords, node);
      double* xyz = stk::mesh::field_data(*currCoords, node);
      double* dx = stk::mesh::field_data(*displacement, node);
      double* velxyz = stk::mesh::field_data(*meshVelocity, node);

      // temporary current and model coords for a generic 2D and 3D implementation
      double mX[3] = {0.0,0.0,0.0};
      double cX[3] = {0.0,0.0,0.0};

      // copy over model coordinates
      for ( int i = 0; i < nDim; ++i )
        mX[i] = oldxyz[i];

      // compute composite transformation matrix
      MotionBase::TransMatType trans_mat = compute_transformation(time,mX);

      // perform matrix multiplication between transformation matrix
      // and old coordinates to obtain current coordinates
      for (int d = 0; d < nDim; d++) {
        xyz[d] = trans_mat[d][0]*mX[0]
                +trans_mat[d][1]*mX[1]
                +trans_mat[d][2]*mX[2]
                +trans_mat[d][3];

        dx[d] = xyz[d] - mX[d];
      } // end for loop - d index

      // copy over current coordinates
      for ( int i = 0; i < nDim; ++i )
        cX[i] = xyz[i];

      // compute velocity vector on current node resulting from all
      // motions in current motion frame
      for (auto& mm: meshMotionVec_)
      {
        MotionBase::ThreeDVecType mm_vel = mm->compute_velocity(time,trans_mat,mX,cX);

        for (int d = 0; d < nDim; d++)
          velxyz[d] += mm_vel[d];
      } // end for loop - mm

    } // end for loop - in index
  } // end for loop - bkts
}

MotionBase::TransMatType FrameNonInertial::compute_transformation(
  const double time,
  const double* xyz)
{
  // all non-inertial frame motions are based off of the reference frame
  MotionBase::TransMatType comp_trans_mat = refFrame_;

  for (auto& mm: meshMotionVec_)
  {
    // build and get transformation matrix
    mm->build_transformation(time,xyz);

    // composite addition of motions in current group
    comp_trans_mat = mm->add_motion(mm->get_trans_mat(),comp_trans_mat);
  }

  return comp_trans_mat;
}

void FrameNonInertial::post_compute_geometry()
{
  // flag denoting if mesh velocity divergence already computed
  bool computedMeshVelDiv = false;

  for (auto& mm: meshMotionVec_)
    mm->post_compute_geometry(bulk_,partVec_,partVecBc_,computedMeshVelDiv);
}

} // nalu
} // sierra
