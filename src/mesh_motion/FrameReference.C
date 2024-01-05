
#include "mesh_motion/FrameReference.h"

#include "FieldTypeDef.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"
#include "stk_mesh/base/GetNgpMesh.hpp"

#include <cassert>

namespace sierra {
namespace nalu {

void
FrameReference::update_coordinates(const double time)
{
  assert(partVec_.size() > 0);

  // create NGP view of motion kernels
  const size_t numKernels = motionKernels_.size();
  auto ngpKernels = nalu_ngp::create_ngp_view<NgpMotion>(motionKernels_);

  // define mesh entities
  const int nDim = meta_.spatial_dimension();
  const auto& ngpMesh = stk::mesh::get_updated_ngp_mesh(bulk_);
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords =
    stk::mesh::get_updated_ngp_field<double>(
      *meta_.get_field<VectorFieldType>(entityRank, "coordinates"));

  // sync fields to device
  modelCoords.sync_to_device();

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::selectUnion(partVec_) &
    (meta_.locally_owned_part() | meta_.globally_shared_part());

  // NGP for loop to update coordinates
  nalu_ngp::run_entity_algorithm(
    "FrameReference_update_coordinates", ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(
      const nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {
      // temporary model coords for a generic 2D and 3D implementation
      mm::ThreeDVecType mX;

      // copy over model coordinates
      for (int d = 0; d < nDim; ++d)
        mX[d] = modelCoords.get(mi, d);

      // initialize composite transformation matrix
      mm::TransMatType compTransMat;

      // create composite transformation matrix based off of all motions
      for (size_t i = 0; i < numKernels; ++i) {
        NgpMotion* kernel = ngpKernels(i);

        // build and get transformation matrix
        mm::TransMatType currTransMat = kernel->build_transformation(time, mX);

        // composite addition of motions in current group
        compTransMat = kernel->add_motion(currTransMat, compTransMat);
      }

      // perform matrix multiplication between transformation matrix
      // and old coordinates to obtain current coordinates
      for (int d = 0; d < nDim; ++d) {
        modelCoords.get(mi, d) = compTransMat[d * mm::matSize + 0] * mX[0] +
                                 compTransMat[d * mm::matSize + 1] * mX[1] +
                                 compTransMat[d * mm::matSize + 2] * mX[2] +
                                 compTransMat[d * mm::matSize + 3];
      } // end for loop - d index
    }); // end NGP for loop

  // Mark fields as modified on device
  modelCoords.modify_on_device();
}

} // namespace nalu
} // namespace sierra
