#include <gtest/gtest.h>

#include <aero/fmb/KynemaFMBBase.h>

#include "UnitTestUtils.h"

#include <algorithm>
#include <cmath>

namespace {

using sierra::kynema_ugf::BeamBody;
using sierra::kynema_ugf::KynemaFMBBase;
using sierra::kynema_ugf::PointMass;
using sierra::kynema_ugf::VectorFieldType;

class TestFMB : public KynemaFMBBase
{
public:
  void setup(double, std::shared_ptr<stk::mesh::BulkData>) override {}
  void initialize(int, double) override {}
  void map_displacements(double, bool) override {}
  void advance_struct_timestep(double, double) override {}
  void map_loads(double) override {}
  stk::mesh::PartVector get_mesh_blocks() const override { return {}; }
};

class KynemaFMBBaseTest : public ::testing::Test
{
protected:
  KynemaFMBBaseTest()
  {
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(3);
    bulk_ = meshBuilder.create();
    meta_ = &bulk_->mesh_meta_data();
    meta_->use_simple_fields();

    currentCoords_ = &meta_->declare_field<double>(
      stk::topology::NODE_RANK, "current_coordinates");
    meshDisp_ = &meta_->declare_field<double>(
      stk::topology::NODE_RANK, "mesh_displacement");
    meshVelocity_ =
      &meta_->declare_field<double>(stk::topology::NODE_RANK, "mesh_velocity");
    beamXi_ =
      &meta_->declare_field<double>(stk::topology::NODE_RANK, "beam_xi");
    pressure_ =
      &meta_->declare_field<double>(stk::topology::NODE_RANK, "pressure");
    density_ =
      &meta_->declare_field<double>(stk::topology::NODE_RANK, "density", 3);
    viscosity_ = &meta_->declare_field<double>(
      stk::topology::NODE_RANK, "effective_viscosity_u");
    dudx_ = &meta_->declare_field<double>(stk::topology::NODE_RANK, "dudx");
    tforce_ = &meta_->declare_field<double>(stk::topology::NODE_RANK, "tforce");
    exposedArea_ =
      &meta_->declare_field<double>(meta_->side_rank(), "exposed_area_vector");
    tforceScs_ =
      &meta_->declare_field<double>(meta_->side_rank(), "tforce_scs");

    const double zero[3] = {0.0, 0.0, 0.0};
    stk::mesh::put_field_on_mesh(
      *currentCoords_, meta_->universal_part(), 3, zero);
    stk::mesh::put_field_on_mesh(*meshDisp_, meta_->universal_part(), 3, zero);
    stk::mesh::put_field_on_mesh(
      *meshVelocity_, meta_->universal_part(), 3, zero);
    stk::mesh::put_field_on_mesh(*beamXi_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*pressure_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*density_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*viscosity_, meta_->universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(*dudx_, meta_->universal_part(), 9, nullptr);
    stk::mesh::put_field_on_mesh(*tforce_, meta_->universal_part(), 3, zero);
    stk::mesh::put_field_on_mesh(
      *exposedArea_, meta_->universal_part(), 12, nullptr);
    stk::mesh::put_field_on_mesh(
      *tforceScs_, meta_->universal_part(), 12, nullptr);

    unit_test_utils::fill_hex8_mesh("generated:1x1x1|sideset:xXyYzZ", *bulk_);
    forcingSurface_ = meta_->get_part("surface_1");
    EXPECT_NE(nullptr, forcingSurface_);

    const auto* coordinates =
      static_cast<const VectorFieldType*>(meta_->coordinate_field());
    for (const auto* bucket : bulk_->get_buckets(
           stk::topology::NODE_RANK, meta_->locally_owned_part())) {
      for (const auto node : *bucket) {
        const double* model = stk::mesh::field_data(*coordinates, node);
        double* current = stk::mesh::field_data(*currentCoords_, node);
        for (int component = 0; component < 3; ++component)
          current[component] = model[component];
        *stk::mesh::field_data(*pressure_, node) = 2.0;
        *stk::mesh::field_data(*density_, node) = 1.0;
        *stk::mesh::field_data(*viscosity_, node) = 0.0;
        std::fill_n(stk::mesh::field_data(*dudx_, node), 9, 0.0);
      }
    }
    for (const auto* bucket : bulk_->get_buckets(
           meta_->side_rank(),
           meta_->locally_owned_part() & *forcingSurface_)) {
      for (const auto face : *bucket) {
        double* area = stk::mesh::field_data(*exposedArea_, face);
        for (int ip = 0; ip < 4; ++ip) {
          area[3 * ip] = 1.0;
          area[3 * ip + 1] = 0.0;
          area[3 * ip + 2] = 0.0;
        }
      }
    }
    currentCoords_->modify_on_host();
    pressure_->modify_on_host();
    density_->modify_on_host();
    viscosity_->modify_on_host();
    dudx_->modify_on_host();
    exposedArea_->modify_on_host();
  }

  PointMass make_point_mass()
  {
    PointMass point;
    point.bulk = bulk_;
    point.moving_mesh_blocks = {&meta_->universal_part()};
    point.forcing_surfaces = {forcingSurface_};
    point.total_force = tforceScs_;
    point.calc_loads =
      std::make_shared<sierra::kynema_ugf::CalcLoads>(point.forcing_surfaces);
    point.calc_loads->setup(bulk_);
    return point;
  }

  BeamBody make_beam_body()
  {
    BeamBody beam;
    beam.bulk = bulk_;
    beam.n_nodes = 2;
    beam.beam_data.resize(beam.n_nodes);
    beam.beam_data_host.resize(beam.n_nodes);
    beam.moving_mesh_blocks = {&meta_->universal_part()};
    beam.forcing_surfaces = {forcingSurface_};
    beam.calc_loads = std::make_shared<sierra::kynema_ugf::CalcLoadsAssembled>(
      beam.forcing_surfaces);
    beam.calc_loads->setup(bulk_);
    beam.beam_data_host.node_xi(0) = 0.0;
    beam.beam_data_host.node_xi(1) = 1.0;
    beam.beam_data_host.pos(0, 0) = 0.0;
    beam.beam_data_host.pos(1, 0) = 1.0;
    for (int node = 0; node < 2; ++node) {
      beam.beam_data_host.pos(node, 1) = 0.0;
      beam.beam_data_host.pos(node, 2) = 0.0;
      beam.beam_data_host.pos(node, 3) = 1.0;
      beam.beam_data_host.pos(node, 4) = 0.0;
      beam.beam_data_host.pos(node, 5) = 0.0;
      beam.beam_data_host.pos(node, 6) = 0.0;
      beam.beam_data_host.disp(node, 0) = 1.0;
      beam.beam_data_host.disp(node, 1) = 2.0;
      beam.beam_data_host.disp(node, 2) = 3.0;
      beam.beam_data_host.disp(node, 3) = std::sqrt(0.5);
      beam.beam_data_host.disp(node, 4) = 0.0;
      beam.beam_data_host.disp(node, 5) = 0.0;
      beam.beam_data_host.disp(node, 6) = std::sqrt(0.5);
      beam.beam_data_host.vel(node, 0) = 4.0;
      beam.beam_data_host.vel(node, 1) = 5.0;
      beam.beam_data_host.vel(node, 2) = 6.0;
      beam.beam_data_host.vel(node, 3) = 0.0;
      beam.beam_data_host.vel(node, 4) = 0.0;
      beam.beam_data_host.vel(node, 5) = 2.0;
    }
    return beam;
  }

  stk::mesh::MetaData* meta_{nullptr};
  std::shared_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Part* forcingSurface_{nullptr};
  VectorFieldType* currentCoords_{nullptr};
  VectorFieldType* meshDisp_{nullptr};
  VectorFieldType* meshVelocity_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* beamXi_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* pressure_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* density_{nullptr};
  sierra::kynema_ugf::ScalarFieldType* viscosity_{nullptr};
  sierra::kynema_ugf::GenericFieldType* dudx_{nullptr};
  VectorFieldType* tforce_{nullptr};
  sierra::kynema_ugf::GenericFieldType* exposedArea_{nullptr};
  sierra::kynema_ugf::GenericFieldType* tforceScs_{nullptr};
  TestFMB fmb_;
};

TEST_F(KynemaFMBBaseTest, MapsPointDisplacementVelocityAndCurrentCoordinates)
{
  auto point = make_point_mass();
  point.p_data.pos = {1.0, 2.0, 3.0, std::sqrt(0.5), 0.0, 0.0, std::sqrt(0.5)};
  point.p_data.vel = {4.0, 5.0, 6.0, 0.0, 0.0, 2.0};
  point.p_data.center_of_mass = {0.0, 0.0, 0.0};

  fmb_.map_displacements_point(point, false);
  const auto* coordinates =
    static_cast<const VectorFieldType*>(meta_->coordinate_field());
  for (const auto* bucket : bulk_->get_buckets(
         stk::topology::NODE_RANK, meta_->locally_owned_part())) {
    for (const auto node : *bucket) {
      const double* model = stk::mesh::field_data(*coordinates, node);
      const double* current = stk::mesh::field_data(*currentCoords_, node);
      EXPECT_DOUBLE_EQ(model[0], current[0]);
      EXPECT_DOUBLE_EQ(model[1], current[1]);
      EXPECT_DOUBLE_EQ(model[2], current[2]);
    }
  }

  fmb_.map_displacements_point(point, true);

  for (const auto* bucket : bulk_->get_buckets(
         stk::topology::NODE_RANK, meta_->locally_owned_part())) {
    for (const auto node : *bucket) {
      const double* model = stk::mesh::field_data(*coordinates, node);
      const double* displacement = stk::mesh::field_data(*meshDisp_, node);
      const double* velocity = stk::mesh::field_data(*meshVelocity_, node);
      const double* current = stk::mesh::field_data(*currentCoords_, node);
      EXPECT_NEAR(1.0 - model[0] - model[1], displacement[0], 1.e-12);
      EXPECT_NEAR(2.0 + model[0] - model[1], displacement[1], 1.e-12);
      EXPECT_NEAR(3.0, displacement[2], 1.e-12);
      EXPECT_NEAR(4.0 - 2.0 * model[0], velocity[0], 1.e-12);
      EXPECT_NEAR(5.0 - 2.0 * model[1], velocity[1], 1.e-12);
      EXPECT_NEAR(6.0, velocity[2], 1.e-12);
      EXPECT_NEAR(model[0] + displacement[0], current[0], 1.e-12);
      EXPECT_NEAR(model[1] + displacement[1], current[1], 1.e-12);
      EXPECT_NEAR(model[2] + displacement[2], current[2], 1.e-12);
    }
  }
}

TEST_F(KynemaFMBBaseTest, MapsBeamDisplacementsAfterComputingBeamCoordinates)
{
  auto beam = make_beam_body();

  fmb_.compute_mapping_beam(beam);
  fmb_.map_displacements_beam(beam, false);
  currentCoords_->sync_to_host();
  const auto* coordinates =
    static_cast<const VectorFieldType*>(meta_->coordinate_field());
  for (const auto* bucket : bulk_->get_buckets(
         stk::topology::NODE_RANK, meta_->locally_owned_part())) {
    for (const auto node : *bucket) {
      const double* model = stk::mesh::field_data(*coordinates, node);
      const double* current = stk::mesh::field_data(*currentCoords_, node);
      EXPECT_DOUBLE_EQ(model[0], current[0]);
      EXPECT_DOUBLE_EQ(model[1], current[1]);
      EXPECT_DOUBLE_EQ(model[2], current[2]);
    }
  }

  fmb_.map_displacements_beam(beam, true);
  beamXi_->sync_to_host();
  meshDisp_->sync_to_host();
  meshVelocity_->sync_to_host();
  currentCoords_->sync_to_host();

  for (const auto* bucket : bulk_->get_buckets(
         stk::topology::NODE_RANK, meta_->locally_owned_part())) {
    for (const auto node : *bucket) {
      const double* model = stk::mesh::field_data(*coordinates, node);
      const double* displacement = stk::mesh::field_data(*meshDisp_, node);
      const double* velocity = stk::mesh::field_data(*meshVelocity_, node);
      const double* current = stk::mesh::field_data(*currentCoords_, node);
      const double xi = *stk::mesh::field_data(*beamXi_, node);
      EXPECT_GE(xi, 0.0);
      EXPECT_LE(xi, 1.0);
      EXPECT_NEAR(1.0 + xi - model[0] - model[1], displacement[0], 1.e-12);
      EXPECT_NEAR(2.0 + model[0] - xi - model[1], displacement[1], 1.e-12);
      EXPECT_NEAR(3.0, displacement[2], 1.e-12);
      EXPECT_NEAR(4.0 - 2.0 * (model[0] - xi), velocity[0], 1.e-12);
      EXPECT_NEAR(5.0 - 2.0 * model[1], velocity[1], 1.e-12);
      EXPECT_NEAR(6.0, velocity[2], 1.e-12);
      EXPECT_NEAR(model[0] + displacement[0], current[0], 1.e-12);
      EXPECT_NEAR(model[1] + displacement[1], current[1], 1.e-12);
      EXPECT_NEAR(model[2] + displacement[2], current[2], 1.e-12);
    }
  }
}

TEST_F(KynemaFMBBaseTest, MapsPointLoadsWithCalcLoads)
{
  auto point = make_point_mass();
  point.p_data.pos = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

  fmb_.map_loads_point(point);

  EXPECT_NEAR(48.0, point.p_data.loads[0], 1.e-12);
  EXPECT_NEAR(0.0, point.p_data.loads[1], 1.e-12);
  EXPECT_NEAR(0.0, point.p_data.loads[2], 1.e-12);
}

TEST_F(KynemaFMBBaseTest, MapsBeamLoadsWithCalcLoadsAssembled)
{
  auto beam = make_beam_body();
  fmb_.compute_mapping_beam(beam);

  fmb_.map_loads_beam(beam);

  double totalForceX = 0.0;
  for (std::size_t node = 0; node < beam.n_nodes; ++node)
    totalForceX += beam.beam_data_host.loads(node, 0);
  EXPECT_NEAR(16.0, totalForceX, 1.e-12);
}

} // namespace