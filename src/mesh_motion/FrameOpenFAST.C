#include "mesh_motion/FrameOpenFAST.h"

// stk_mesh/base/fem
#include <stk_mesh/base/FieldBLAS.hpp>

#include <cassert>

namespace sierra{
namespace nalu{

void FrameOpenFAST::update_coordinates_velocity(const double time)
{
  const int ndim = meta_.spatial_dimension();

  if (fsiTurbineData_ != NULL) {
      fsiTurbineData_->mapDisplacements();

      VectorFieldType* modelCoords = meta_.get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "coordinates");
      VectorFieldType* currCoords = meta_.get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "current_coordinates");
      VectorFieldType* displacement = meta_.get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "mesh_displacement");
      //VectorFieldType &displacementN = displacement->field_of_state(stk::mesh::StateN);
      //VectorFieldType &displacementNp1 = displacement->field_of_state(stk::mesh::StateNP1);

      VectorFieldType* meshVelocity = meta_.get_field<VectorFieldType>(
          stk::topology::NODE_RANK, "mesh_velocity");

      // get the parts in the current motion frame
      stk::mesh::Selector sel = stk::mesh::selectUnion(fsiTurbineData_->getPartVec());
      const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

      for (auto b: bkts) {
          for (size_t in=0; in < b->size(); in++) {

              auto node = (*b)[in]; // mesh node and NOT YAML node
              double* oldxyz = stk::mesh::field_data(*modelCoords, node);
              double* xyz = stk::mesh::field_data(*currCoords, node);
              double* dxNp1 = stk::mesh::field_data(*displacement, node);
              //double* dxN = stk::mesh::field_data(displacementN, node);
              double* velxyz = stk::mesh::field_data(*meshVelocity, node);

              // perform matrix multiplication between transformation matrix
              // and old coordinates to obtain current coordinates
              for (int d = 0; d < ndim; d++) {
                  xyz[d] = oldxyz[d] + dxNp1[d];
                  // velxyz[d] =  (dxNp1[d] - dxN[d])/deltaT;
              } // end for loop - d index

          } // end for loop - in index
      } // end for loop - bkts
  }
}

} // nalu
} // sierra
