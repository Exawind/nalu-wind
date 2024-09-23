#include <gtest/gtest.h>
#include <limits>

#include "mesh_motion/MeshTransformationAlg.h"
#include "mesh_motion/MotionRotationKernel.h"
#include "mesh_motion/MotionScalingKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

#include <string>

namespace {
const double testTol = 1e-12;

// create yaml nodes describing mesh motion
const std::string mInfo = "mesh_transformation:               \n"
                          "  - name: rotate                   \n"
                          "    mesh_parts: [ all_blocks ]     \n"
                          "    compute_centroid: true         \n"
                          "    motion:                        \n"
                          "     - type: rotation              \n"
                          "       angle: 90                   \n"
                          "       axis: [0, 0, 1]             \n"
                          "                                   \n"
                          "     - type: scaling               \n"
                          "       factor: [1.2, 1.0, 1.2]     \n";
// define YAML node
const YAML::Node yamlNode = YAML::Load(mInfo);
const YAML::Node mesh_transformation = yamlNode["mesh_transformation"];

const std::string mInfoGold = "mesh_transformation:               \n"
                              "  - name: rotate                   \n"
                              "    mesh_parts: [ all_blocks ]     \n"
                              "    motion:                        \n"
                              "     - type: rotation              \n"
                              "       angle: 90                   \n"
                              "       axis: [0, 0, 1]             \n"
                              "       centroid: [2.5, 4.5, 5.5]   \n"
                              "                                   \n"
                              "     - type: scaling               \n"
                              "       factor: [1.2, 1.0, 1.2]     \n"
                              "       centroid: [2.5, 4.5, 5.5]   \n";
// define YAML node
const YAML::Node yamlNodeGold = YAML::Load(mInfoGold);
const YAML::Node mesh_transformation_gold = yamlNodeGold["mesh_transformation"];
const YAML::Node rot_scl = mesh_transformation_gold[0]["motion"];
const YAML::Node rotNode = rot_scl[0];
const YAML::Node scaleNode = rot_scl[1];

sierra::nalu::mm::TransMatType
eval_transformation(sierra::nalu::Realm& realm, double time, const double* xyz)
{
  // transform data structures to confirm to mesh motion
  sierra::nalu::mm::ThreeDVecType vecX;
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++)
    vecX[d] = xyz[d];

  // perform 1st rotation transformation
  sierra::nalu::MotionRotationKernel rotClass(rotNode);
  sierra::nalu::mm::TransMatType compTrans =
    rotClass.build_transformation(time, vecX);

  // perform 2nd srotation transformation
  sierra::nalu::MotionScalingKernel scaleClass(realm.meta_data(), scaleNode);
  sierra::nalu::mm::TransMatType tempMat =
    scaleClass.build_transformation(time, vecX);
  return scaleClass.add_motion(tempMat, compTrans);
}

std::vector<double>
eval_coords(const sierra::nalu::mm::TransMatType& transMat, const double* xyz)
{
  std::vector<double> transCoord(3, 0.0);

  // perform matrix multiplication between transformation matrix
  // and original coordinates to obtain transformed coordinates
  for (int d = 0; d < sierra::nalu::nalu_ngp::NDimMax; d++) {
    transCoord[d] = transMat[d * sierra::nalu::mm::matSize + 0] * xyz[0] +
                    transMat[d * sierra::nalu::mm::matSize + 1] * xyz[1] +
                    transMat[d * sierra::nalu::mm::matSize + 2] * xyz[2] +
                    transMat[d * sierra::nalu::mm::matSize + 3];
  }

  return transCoord;
}
} // namespace

TEST(meshMotion, NGP_compute_centroid)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  realm.solutionOptions_->meshTransformation_ = true;

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  realm.timeIntegrator_ = &timeIntegrator;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields(
    stk::mesh::PartVector(1, &(realm.meta_data().universal_part())));

  // create field to copy coordinates
  // NOTE: This is done to allow computation of gold values later on
  // because mesh_transformation changes the field - coordinates
  int nDim = realm.meta_data().spatial_dimension();
  sierra::nalu::VectorFieldType* modelCoordsGold =
    &(realm.meta_data().declare_field<double>(
      stk::topology::NODE_RANK, "coordinates_gold"));
  stk::mesh::put_field_on_mesh(
    *modelCoordsGold, realm.meta_data().universal_part(), nDim, nullptr);
  stk::io::set_field_output_type(
    *modelCoordsGold, stk::io::FieldOutputType::VECTOR_3D);

  // create mesh and get dimensions
  const std::string meshSpec("generated:5x9x11");
  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::Selector(realm.meta_data().universal_part()) &
    (realm.meta_data().locally_owned_part() |
     realm.meta_data().globally_shared_part());
  const auto& bkts =
    realm.bulk_data().get_buckets(stk::topology::NODE_RANK, sel);

  // get model coordinate fields
  sierra::nalu::VectorFieldType* modelCoords =
    realm.meta_data().get_field<double>(
      stk::topology::NODE_RANK, "coordinates");

  // copy over coordinates
  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* mxyz = stk::mesh::field_data(*modelCoords, node);
      double* gxyz = stk::mesh::field_data(*modelCoordsGold, node);

      for (int d = 0; d < nDim; ++d)
        gxyz[d] = mxyz[d];
    } // end for loop - in index
  } // end for loop - bkts

  // create mesh transformation algorithm class
  sierra::nalu::MeshTransformationAlg meshTransformationAlg(
    realm.bulk_data(), mesh_transformation);
  const double time = 0.0;
  meshTransformationAlg.initialize(time);

  // sync coordinates to host
  modelCoords->sync_to_host();

  // compare coordinates
  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in]; // mesh node and NOT YAML node
      double* gxyz = stk::mesh::field_data(*modelCoordsGold, node);
      double* mxyz = stk::mesh::field_data(*modelCoords, node);

      sierra::nalu::mm::TransMatType transMat =
        eval_transformation(realm, time, gxyz);

      std::vector<double> gold_norm_xyz = eval_coords(transMat, gxyz);

      EXPECT_NEAR(mxyz[0], gold_norm_xyz[0], testTol);
      EXPECT_NEAR(mxyz[1], gold_norm_xyz[1], testTol);
      EXPECT_NEAR(mxyz[2], gold_norm_xyz[2], testTol);
    } // end for loop - in index
  } // end for loop - bkts
}
