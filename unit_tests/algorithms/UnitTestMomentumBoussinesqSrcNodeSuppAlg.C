// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "UnitTestUtils.h"
#include "UnitTestAlgorithm.h"
#include "UnitTestAlgorithmUtils.h"
#include "UnitTestRealm.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "MomentumBoussinesqRASrcNodeSuppAlg.h"
#include "MovingAveragePostProcessor.h"
#include "TimeIntegrator.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

TEST(TestMomentumBoussinesqRASrcNodeSuppAlg, single_value)
{
  NodeSuppHelper helper;
  auto& meta = helper.realm.meta_data();
  auto& bulk = helper.realm.bulk_data();

  auto& dnv =
    meta.declare_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
  stk::mesh::put_field_on_mesh(dnv, meta.universal_part(), nullptr);

  auto& temperature =
    meta.declare_field<double>(stk::topology::NODE_RANK, "temperature");
  stk::mesh::put_field_on_mesh(temperature, meta.universal_part(), nullptr);

  std::string avgTempFieldName =
    sierra::nalu::MovingAveragePostProcessor::filtered_field_name(
      "temperature");
  auto& raTemperature =
    meta.declare_field<double>(stk::topology::NODE_RANK, avgTempFieldName);
  stk::mesh::put_field_on_mesh(raTemperature, meta.universal_part(), nullptr);

  meta.commit();

  stk::mesh::Entity node = helper.make_one_node_mesh();

  const double dnv_value = 0.125;
  *stk::mesh::field_data(dnv, node) = dnv_value;

  const double temperature_value = 300;
  *stk::mesh::field_data(temperature, node) = temperature_value;

  auto& solnOpts = *helper.realm.solutionOptions_;

  const double rhoRef = 1;
  solnOpts.referenceDensity_ = rhoRef;

  const double beta = 1.0 / 300.0;
  solnOpts.thermalExpansionCoeff_ = beta;

  std::vector<double> gravity = {-5, 6, 7};
  solnOpts.gravity_ = gravity;

  const double timeScale = 2;
  solnOpts.raBoussinesqTimeScale_ = timeScale;

  sierra::nalu::TimeIntegrator timeIntg;
  timeIntg.currentTime_ = 0;
  timeIntg.timeStepN_ = 1;
  sierra::nalu::MovingAveragePostProcessor avgPP(bulk, timeIntg, false);
  avgPP.add_fields({"temperature"});
  avgPP.set_time_scale(solnOpts.raBoussinesqTimeScale_);
  avgPP.execute();

  double rhs[3] = {0, 0, 0};
  auto boussinesqRaAlg =
    sierra::nalu::MomentumBoussinesqRASrcNodeSuppAlg(helper.realm);
  boussinesqRaAlg.setup();
  boussinesqRaAlg.node_execute(nullptr, rhs, node);

  for (int d = 0; d < 3; ++d) {
    EXPECT_DOUBLE_EQ(rhs[d], 0.0);
  }

  const double temperature_value_new = 305;
  *stk::mesh::field_data(temperature, node) = temperature_value_new;

  avgPP.execute();

  double alpha = timeIntg.timeStepN_ / timeScale;
  const double avgTempVal =
    alpha * temperature_value_new + (1 - alpha) * temperature_value;
  EXPECT_DOUBLE_EQ(avgTempVal, 302.5);

  double coeff =
    -beta * rhoRef * (temperature_value_new - avgTempVal) * dnv_value;
  double expected_rhs[3] = {
    coeff * gravity[0], coeff * gravity[1], coeff * gravity[2]};

  double newRHS[3] = {0, 0, 0};
  boussinesqRaAlg.node_execute(nullptr, newRHS, node);

  for (int d = 0; d < 3; ++d) {
    EXPECT_DOUBLE_EQ(newRHS[d], expected_rhs[d]);
  }
}
