// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <limits>
#include <stk_mesh/base/CreateEdges.hpp>
#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "mesh_motion/MeshMotionAlg.h"
#include "ngp_algorithms/GeometryInteriorAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/WallFuncGeometryAlg.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "gcl/MeshVelocityAlg.h"
#include "gcl/MeshVelocityEdgeAlg.h"
#include "utils/StkHelpers.h"
#include "utils/ComputeVectorDivergence.h"

namespace {

class GCLTest : public ::testing::Test
{
public:
  GCLTest()
    : naluObj_(),
      realm_(naluObj_.create_realm()),
      meta_(realm_.meta_data()),
      bulk_(realm_.bulk_data()),
      geomAlgDriver_(realm_),
      currCoords_(&meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "current_coordinates", numStates_)),
      dualVol_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "dual_nodal_volume", numStates_)),
      elemVol_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::ELEM_RANK, "element_volume")),
      edgeAreaVec_(&meta_.declare_field<VectorFieldType>(
        stk::topology::EDGE_RANK, "edge_area_vector")),
      exposedAreaVec_(&meta_.declare_field<GenericFieldType>(
        meta_.side_rank(), "exposed_area_vector")),
      meshDisp_(&meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_displacement", numStates_)),
      meshVel_(&meta_.declare_field<VectorFieldType>(
        stk::topology::NODE_RANK, "mesh_velocity", numStates_)),
      sweptVol_(&(meta_.declare_field<GenericFieldType>(
        stk::topology::ELEM_RANK, "swept_face_volume", numStates_))),
      faceVelMag_(&(meta_.declare_field<GenericFieldType>(
        stk::topology::ELEM_RANK, "face_velocity_mag", numStates_))),
      edgeSweptVol_(&(meta_.declare_field<GenericFieldType>(
        stk::topology::EDGE_RANK, "edge_swept_face_volume", numStates_))),
      edgeFaceVelMag_(&(meta_.declare_field<GenericFieldType>(
        stk::topology::EDGE_RANK, "edge_face_velocity_mag", numStates_))),
      divMeshVel_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "div_mesh_velocity")),
      dVoldt_(&meta_.declare_field<ScalarFieldType>(
        stk::topology::NODE_RANK, "dvol_dt"))
  {
    realm_.timeIntegrator_ = naluObj_.sim_.timeIntegrator_;
    stk::mesh::put_field_on_mesh(
      *currCoords_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(*dualVol_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*elemVol_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(
      *edgeAreaVec_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *exposedAreaVec_, meta_.universal_part(),
      spatialDim_ * sierra::nalu::AlgTraitsQuad4::numScsIp_, nullptr);
    stk::mesh::put_field_on_mesh(
      *meshDisp_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *meshVel_, meta_.universal_part(), spatialDim_, nullptr);
    stk::mesh::put_field_on_mesh(
      *sweptVol_, meta_.universal_part(),
      sierra::nalu::AlgTraitsHex8::numScsIp_, nullptr);
    stk::mesh::put_field_on_mesh(
      *faceVelMag_, meta_.universal_part(),
      sierra::nalu::AlgTraitsHex8::numScsIp_, nullptr);
    stk::mesh::put_field_on_mesh(
      *edgeSweptVol_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(
      *edgeFaceVelMag_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(
      *divMeshVel_, meta_.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*dVoldt_, meta_.universal_part(), 1, nullptr);
  }

  virtual ~GCLTest() = default;

  void fill_mesh_and_init_fields(
    const std::string meshSize,
    bool doPerturb = true,
    bool generateSidesets = true)
  {
    std::string meshSpec = "generated:" + meshSize;
    if (generateSidesets)
      meshSpec += "|sideset:xXyYzZ";
    unit_test_utils::fill_hex8_mesh(meshSpec, bulk_);
    if (doPerturb)
      unit_test_utils::perturb_coord_hex_8(bulk_);

    partVec_ = {meta_.get_part("block_1")};
    coordinates_ =
      static_cast<const VectorFieldType*>(meta_.coordinate_field());
    EXPECT_TRUE(coordinates_ != nullptr);

    stk::mesh::create_edges(bulk_, meta_.universal_part());
    bndyPartVec_ = {meta_.get_part("surface_1")};
  }

  void init_time_integrator(
    bool secondOrder = true, double timeStep = 0.1, int timeStepCount = 2)
  {
    auto& timeInt = *realm_.timeIntegrator_;
    timeInt.secondOrderTimeAccurate_ = secondOrder;
    if (!secondOrder)
      numStates_ = 2;
    timeInt.timeStepN_ = timeStep;
    timeInt.timeStepNm1_ = timeStep;
    timeInt.timeStepCount_ = timeStepCount;
    if (timeInt.secondOrderTimeAccurate_)
      timeInt.compute_gamma();
  }

  void register_algorithms(const std::string& motion_options)
  {
    // Force mesh motion logic everywhere
    realm_.solutionOptions_->meshMotion_ = true;
    const YAML::Node motionNode = YAML::Load(motion_options);
    realm_.meshMotionAlg_.reset(
      new sierra::nalu::MeshMotionAlg(bulk_, motionNode["mesh_motion"]));

    geomAlgDriver_.register_elem_algorithm<sierra::nalu::GeometryInteriorAlg>(
      sierra::nalu::INTERIOR, partVec_[0], "geometry");
    if (realm_.realmUsesEdges_) {
      geomAlgDriver_.register_elem_algorithm<sierra::nalu::MeshVelocityEdgeAlg>(
        sierra::nalu::INTERIOR, partVec_[0], "mesh_vel");
    } else {
      geomAlgDriver_.register_elem_algorithm<sierra::nalu::MeshVelocityAlg>(
        sierra::nalu::INTERIOR, partVec_[0], "mesh_vel");
    }

    auto* part = meta_.get_part("surface_1");
    for (auto* surfPart : part->subsets()) {
      geomAlgDriver_.register_face_algorithm<sierra::nalu::GeometryBoundaryAlg>(
        sierra::nalu::BOUNDARY, surfPart, "geometry");
    }
  }

  void compute_mesh_velocity()
  {
    const stk::mesh::Selector sel = meta_.universal_part();
    const double dt = realm_.get_time_step();
    const double gamma1 = realm_.get_gamma1();
    const double gamma2 = realm_.get_gamma2();
    const double gamma3 = realm_.get_gamma3();

    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    const auto& dispNm1 = meshDisp_->field_of_state(stk::mesh::StateNM1);
    const auto& dispN = meshDisp_->field_of_state(stk::mesh::StateN);
    const auto& dispNp1 = meshDisp_->field_of_state(stk::mesh::StateNP1);

    for (auto* b : bkts) {
      const double* dxNm1 = stk::mesh::field_data(dispNm1, *b);
      const double* dxN = stk::mesh::field_data(dispN, *b);
      const double* dxNp1 = stk::mesh::field_data(dispNp1, *b);
      double* mVel = stk::mesh::field_data(*meshVel_, *b);

      for (size_t in = 0; in < b->size(); ++in) {
        size_t offset = in * spatialDim_;
        for (unsigned d = 0; d < spatialDim_; ++d)
          mVel[offset + d] =
            (gamma1 * dxNp1[offset + d] + gamma2 * dxN[offset + d] +
             gamma3 * dxNm1[offset + d]) /
            dt;
      }
    }
  }

  void compute_div_mesh_vel()
  {
    if (realm_.realmUsesEdges_) {
      sierra::nalu::compute_edge_scalar_divergence(
        bulk_, partVec_, bndyPartVec_, edgeFaceVelMag_, divMeshVel_);
    } else {
      sierra::nalu::compute_scalar_divergence(
        bulk_, partVec_, bndyPartVec_, faceVelMag_, divMeshVel_);
    }
  }

  void compute_dvoldt()
  {
    const stk::mesh::Selector sel = get_selector();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    const auto& dVolNm1 = dualVol_->field_of_state(stk::mesh::StateNM1);
    const auto& dVolN = dualVol_->field_of_state(stk::mesh::StateN);
    const auto& dVolNp1 = dualVol_->field_of_state(stk::mesh::StateNP1);
    const double dt = realm_.get_time_step();
    const double gamma1 = realm_.get_gamma1();
    const double gamma2 = realm_.get_gamma2();
    const double gamma3 = realm_.get_gamma3();

    for (auto* b : bkts) {
      const double* dvNm1 = stk::mesh::field_data(dVolNm1, *b);
      const double* dvN = stk::mesh::field_data(dVolN, *b);
      const double* dvNp1 = stk::mesh::field_data(dVolNp1, *b);
      double* dvdt = stk::mesh::field_data(*dVoldt_, *b);

      for (size_t in = 0; in < b->size(); ++in) {
        dvdt[in] =
          (gamma1 * dvNp1[in] + gamma2 * dvN[in] + gamma3 * dvNm1[in]) / dt;
      }
    }
  }

  void compute_absolute_error()
  {
    const double tol = 1e-13;
    const stk::mesh::Selector sel = get_selector();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
    const double dt = realm_.get_time_step();

    for (auto* b : bkts) {
      const double* divV = stk::mesh::field_data(*divMeshVel_, *b);
      const double* dvdt = stk::mesh::field_data(*dVoldt_, *b);

      for (size_t in = 0; in < b->size(); in++) {
        const double err = (dvdt[in] - divV[in]) * dt;
        EXPECT_NEAR(err, 0.0, tol);
      }
    }
  }

  void compute_relative_error()
  {
    // The value of tolerances are not clear.
    // This needs more investigation for real use cases.
    const double tol = 1e-11;
    const stk::mesh::Selector sel = get_selector();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (auto* b : bkts) {
      const double* divV = stk::mesh::field_data(*divMeshVel_, *b);
      const double* dvdt = stk::mesh::field_data(*dVoldt_, *b);

      for (size_t in = 0; in < b->size(); in++) {
        const double err = (dvdt[in] - divV[in]) / dvdt[in];
        EXPECT_NEAR(err, 0.0, tol);
      }
    }
  }

  /** Selector to loop over interior nodes only
   */
  stk::mesh::Selector get_selector()
  {
    return (meta_.universal_part() & !stk::mesh::selectUnion(bndyPartVec_));
  }

  void init_states()
  {
    const double deltaT = realm_.get_time_step();
    auto& motionAlg = *realm_.meshMotionAlg_;
    motionAlg.initialize(0.0);
    for (int it = 0; it < numStates_; ++it) {
      realm_.swap_states();
      motionAlg.execute(it * deltaT);
      geomAlgDriver_.execute();
    }
  }

  YAML::Node doc_;
  YAML::Node realmNode_;
  unit_test_utils::NaluTest naluObj_;
  sierra::nalu::Realm& realm_;

  int numStates_{3};
  const unsigned spatialDim_{3};
  stk::mesh::MetaData& meta_;
  stk::mesh::BulkData& bulk_;
  sierra::nalu::GeometryAlgDriver geomAlgDriver_;
  stk::mesh::PartVector partVec_;
  stk::mesh::PartVector bndyPartVec_;

  const VectorFieldType* coordinates_{nullptr};
  VectorFieldType* currCoords_{nullptr};
  ScalarFieldType* dualVol_{nullptr};
  ScalarFieldType* elemVol_{nullptr};
  VectorFieldType* edgeAreaVec_{nullptr};
  GenericFieldType* exposedAreaVec_{nullptr};
  VectorFieldType* meshDisp_{nullptr};
  VectorFieldType* meshVel_{nullptr};
  GenericFieldType* sweptVol_{nullptr};
  GenericFieldType* faceVelMag_{nullptr};
  GenericFieldType* edgeSweptVol_{nullptr};
  GenericFieldType* edgeFaceVelMag_{nullptr};
  ScalarFieldType* divMeshVel_{nullptr};
  ScalarFieldType* dVoldt_{nullptr};
};

} // namespace
