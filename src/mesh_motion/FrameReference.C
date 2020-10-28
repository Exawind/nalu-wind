
#include "../../include/mesh_motion/FrameReference.h"

#include "FieldTypeDef.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cassert>

namespace sierra{
namespace nalu{

void FrameReference::update_coordinates(const double time)
{
  assert (partVec_.size() > 0);

  const int nDim = meta_.spatial_dimension();

  VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "coordinates");

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_) &
      (meta_.locally_owned_part() | meta_.globally_shared_part());
  const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {

      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* xyz = stk::mesh::field_data(*modelCoords, node);

      // temporary model coords for a generic 2D and 3D implementation
      double mX[3] = {0.0,0.0,0.0};

      // copy over model coordinates
      for ( int i = 0; i < nDim; ++i )
        mX[i] = xyz[i];

      // all frame motions are based off of the reference frame
      MotionBase::TransMatType comp_trans_mat = MotionBase::identityMat_;
      for (auto& mm: meshMotionVec_)
      {
        // build and get transformation matrix
        mm->build_transformation(time,mX);
        // composite addition of motions in current group
        comp_trans_mat = mm->add_motion(mm->get_trans_mat(),comp_trans_mat);
      }

      // perform matrix multiplication between transformation matrix
      // and old coordinates to obtain current coordinates
      for (int d = 0; d < nDim; d++) {
        xyz[d] = comp_trans_mat[d][0]*mX[0]
                +comp_trans_mat[d][1]*mX[1]
                +comp_trans_mat[d][2]*mX[2]
                +comp_trans_mat[d][3];
      } // end for loop - d index

    } // end for loop - in index
  } // end for loop - bkts
}

} // nalu
} // sierra
