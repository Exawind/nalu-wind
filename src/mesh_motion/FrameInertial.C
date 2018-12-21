
#include "mesh_motion/FrameInertial.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cassert>

namespace sierra{
namespace nalu{

void FrameInertial::update_coordinates_velocity(const double time)
{
  compute_transformation(time);

  // check if any parts have been associated with current frame
  if (partNamesVec_.size() == 0)
    return;

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

  // reset/initialize to 0, velocity field
  stk::mesh::field_fill(0.0, *meshVelocity, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* oldxyz = stk::mesh::field_data(*modelCoords, node);
      double* xyz = stk::mesh::field_data(*currCoords, node);
      double* dx = stk::mesh::field_data(*displacement, node);

      // perform matrix multiplication between transformation matrix
      // and old coordinates to obtain current coordinates
      for (int d = 0; d < ndim; d++) {
        xyz[d] = inertialFrame_[d][0]*oldxyz[0]
                +inertialFrame_[d][1]*oldxyz[1]
                +inertialFrame_[d][2]*oldxyz[2]
                +inertialFrame_[d][3];

        dx[d] = xyz[d] - oldxyz[d];
      } // end for loop - d index

    } // end for loop - in index
  } // end for loop - bkts
}

void FrameInertial::compute_transformation(const double time)
{
  for (auto& mm: meshMotionVec_)
  {
    // build and get transformation matrix
    mm->build_transformation(time);

    // composite addition of motions in current group
    inertialFrame_ = mm->add_motion(mm->get_trans_mat(),inertialFrame_);
  }
}

} // nalu
} // sierra
