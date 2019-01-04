
#include "mesh_motion/FrameNonInertial.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cassert>

namespace sierra{
namespace nalu{

void FrameNonInertial::update_coordinates_velocity(const double time)
{
  assert (partVec_.size() > 0);

  const int ndim = meta_.spatial_dimension();

  VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* currCoords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* displacement = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");
  VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity");

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_);
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

      // compute composite transformation matrix
      MotionBase::transMatType trans_mat = compute_transformation(time,oldxyz);

      // perform matrix multiplication between transformation matrix
      // and old coordinates to obtain current coordinates
      for (int d = 0; d < ndim; d++) {
        xyz[d] = trans_mat[d][0]*oldxyz[0]
                +trans_mat[d][1]*oldxyz[1]
                +trans_mat[d][2]*oldxyz[2]
                +trans_mat[d][3];

        dx[d] = xyz[d] - oldxyz[d];
      } // end for loop - d index

      // compute velocity vector on current node resulting from all
      // motions in current motion frame
      for (auto& mm: meshMotionVec_)
      {
        MotionBase::threeDVecType mm_vel = mm->compute_velocity(time,trans_mat,xyz);

        for (int d = 0; d < ndim; d++)
          velxyz[d] += mm_vel[d];
      } // end for loop - mm

    } // end for loop - in index
  } // end for loop - bkts

}

MotionBase::transMatType FrameNonInertial::compute_transformation(
  const double time,
  const double* xyz)
{
  // all non-inertial frame motions are based off of the reference frame
  MotionBase::transMatType comp_trans_mat = refFrame_;

  for (auto& mm: meshMotionVec_)
  {
    // build and get transformation matrix
    mm->build_transformation(time,xyz);

    // composite addition of motions in current group
    comp_trans_mat = mm->add_motion(mm->get_trans_mat(),comp_trans_mat);
  }

  return comp_trans_mat;
}

} // nalu
} // sierra
