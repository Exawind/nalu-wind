
#include "mesh_motion/FrameReference.h"

#include "FieldTypeDef.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"

#include <cassert>

namespace sierra{
namespace nalu{

void FrameReference::update_coordinates(const double time)
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

  // get the parts in the current motion frame
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec_) &
      (meta_.locally_owned_part() | meta_.globally_shared_part());

  // NGP for loop to update coordinates
  nalu_ngp::run_entity_algorithm(
    "FrameReference_update_coordinates",
    ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {

    // temporary model coords for a generic 2D and 3D implementation
    double mX[3] = {0.0,0.0,0.0};

    // copy over model coordinates
    for (int d = 0; d < nDim; ++d)
      mX[d] = modelCoords.get(mi,d);

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
      modelCoords.get(mi,d) = comp_trans_mat[d][0]*mX[0]
                             +comp_trans_mat[d][1]*mX[1]
                             +comp_trans_mat[d][2]*mX[2]
                             +comp_trans_mat[d][3];
    } // end for loop - d index
  }); // end NGP for loop
}

} // nalu
} // sierra
