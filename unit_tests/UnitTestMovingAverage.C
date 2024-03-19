#include <gtest/gtest.h>
#include <limits>

#include <TimeIntegrator.h>
#include <MovingAveragePostProcessor.h>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>

#include <random>
#include <fstream>

#include "UnitTestUtils.h"

namespace {

class PostProcessor : public ::testing::Test
{
public:
  PostProcessor() : timeIntegrator_(), numSteps(2000)
  {
    auto numberOfDimensions = 2u;
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(numberOfDimensions);
    bulk_ = meshBuilder.create();
    auto& meta_ = bulk_->mesh_meta_data();
    meta_.use_simple_fields();

    temperature_ =
      &meta_.declare_field<double>(stk::topology::NODE_RANK, "temperature");

    raTemperature_ = &meta_.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::MovingAveragePostProcessor::filtered_field_name(
        "temperature"));
    stk::mesh::put_field_on_mesh(
      *temperature_, meta_.universal_part(), nullptr);
    stk::mesh::put_field_on_mesh(
      *raTemperature_, meta_.universal_part(), nullptr);
    meta_.commit();

    bulk_->modification_begin();
    node = bulk_->declare_node(1u);
    bulk_->modification_end();

    const double final_time = 2.0 * std::acos(-1.0);
    timeIntegrator_.timeStepN_ = final_time / (numSteps);
  }

  sierra::nalu::TimeIntegrator timeIntegrator_;
  std::shared_ptr<stk::mesh::BulkData> bulk_;
  int numSteps;

  stk::mesh::Entity node;
  sierra::nalu::ScalarFieldType* temperature_;
  sierra::nalu::ScalarFieldType* raTemperature_;
};

} // namespace

TEST_F(PostProcessor, moving_average_constant)
{
  std::vector<double> constant_realization(numSteps, 10.0);

  double timeScale = 0.1;
  sierra::nalu::MovingAveragePostProcessor avgPP(
    *bulk_, timeIntegrator_, false);
  avgPP.add_fields({"temperature"});
  avgPP.set_time_scale(timeScale);

  for (int j = 0; j < numSteps; ++j) {
    double* temperatureVal = stk::mesh::field_data(*temperature_, node);
    *temperatureVal = constant_realization[j];
    EXPECT_NO_THROW(avgPP.execute());
    double* raTemperatureVal = stk::mesh::field_data(*raTemperature_, node);
    EXPECT_NEAR(*temperatureVal, *raTemperatureVal, 1.0e-10);
  }
}

namespace {
std::vector<double>
ou_realization(double initValue, double dt, int numSteps)
{
  // dx = (mu-x) dt + \sigma dW

  std::mt19937 rng;
  rng.seed(std::random_device()()); // fixed seed
  std::normal_distribution<double> normal(0.0, 1.0);

  std::vector<double> output(numSteps, 0);
  const double tau = 0.01;
  double sigma = 0.1;

  double omega = 1.0;
  auto mean = [omega](double time) { return std::cos(omega * time); };
  output[0] = initValue;
  for (int j = 1; j < numSteps; ++j) {
    double time = j * dt;
    const double dX = (mean(time) - output[j - 1]) * dt / tau +
                      sigma * std::sqrt(dt / tau) * normal(rng);
    output[j] = output[j - 1] + dX;
  }

  return output;
}
} // namespace

TEST_F(PostProcessor, moving_average_ou)
{
  double dt = timeIntegrator_.get_time_step();
  auto realization = ou_realization(1.0, dt, numSteps);

  double timeScale = 0.1;
  sierra::nalu::MovingAveragePostProcessor avgPP(
    *bulk_, timeIntegrator_, false);
  avgPP.add_fields({"temperature"});
  avgPP.set_time_scale(timeScale);

  std::string outputFileName("PostProcessor.moving_average_ou.txt");
  std::ofstream outputFile(outputFileName);
  outputFile << "t, temperature, temperature_avg" << std::endl;
  for (int j = 0; j < numSteps; ++j) {
    *stk::mesh::field_data(*temperature_, node) = realization[j];
    EXPECT_NO_THROW(avgPP.execute());

    outputFile << dt * j << ", " << *stk::mesh::field_data(*temperature_, node)
               << ", " << *stk::mesh::field_data(*raTemperature_, node)
               << std::endl;
  }
  unlink(outputFileName.c_str());
}
