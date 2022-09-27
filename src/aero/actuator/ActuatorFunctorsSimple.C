// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorFunctorsSimple.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <NaluEnv.h>
#include <FieldTypeDef.h>
#include "utils/LinearInterpolation.h"
#include <cmath>
#include <string>
#include <ostream>

namespace sierra {
namespace nalu {

InterpActuatorDensity::InterpActuatorDensity(
  ActuatorBulkSimple& actBulk, stk::mesh::BulkData& stkBulk)
  : actBulk_(actBulk),
    stkBulk_(stkBulk),
    coordinates_(stkBulk_.mesh_meta_data().get_field<VectorFieldType>(
      stk::topology::NODE_RANK, "coordinates")),
    density_(stkBulk_.mesh_meta_data().get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "density"))
{
  actBulk_.density_.sync_host();
  actBulk_.density_.modify_host();
}

void
InterpActuatorDensity::operator()(int index) const
{
  auto rho = actBulk_.density_.view_host();
  auto localCoord = actBulk_.localCoords_;

  if (actBulk_.pointIsLocal_(index)) {

    stk::mesh::Entity elem = stkBulk_.get_entity(
      stk::topology::ELEMENT_RANK, actBulk_.elemContainingPoint_(index));

    const int nodesPerElem = stkBulk_.num_nodes(elem);

    // just allocate for largest expected size (hex27)
    double ws_coordinates[81], ws_density[81];

    // Check to make sure the size is sufficient
    ThrowAssert(81 >= 3 * nodesPerElem);

    actuator_utils::gather_field(
      3, &ws_coordinates[0], *coordinates_, stkBulk_.begin_nodes(elem),
      nodesPerElem);

    actuator_utils::gather_field_for_interp(
      1, &ws_density[0], *density_, stkBulk_.begin_nodes(elem), nodesPerElem);

    actuator_utils::interpolate_field(
      1, elem, stkBulk_, &(localCoord(index, 0)), &ws_density[0],
      &(rho(index)));
    rho(index) /= actBulk_.localParallelRedundancy_(index);
  }
}

#ifdef ENABLE_ACTSIMPLE_PTMOTION
ActSimpleUpdatePoints::ActSimpleUpdatePoints(
  ActuatorBulkSimple& actBulk, int numpts, double p1[], double p2[])
  : points_(helper_.get_local_view(actBulk.pointCentroid_)),
    offsets_(helper_.get_local_view(actBulk.turbIdOffset_)),
    turbId_(actBulk.localTurbineId_),
    numpoints_(numpts)
{
  for (int i = 0; i < 3; i++) {
    p1_[i] = p1[i];
    p2_[i] = p2[i];
  }
  helper_.touch_dual_view(actBulk.pointCentroid_);
}

void
ActSimpleUpdatePoints::operator()(int index) const
{

  ThrowAssert(turbId_ >= 0);
  const int pointId = index - offsets_(turbId_);
  auto point = Kokkos::subview(points_, index, Kokkos::ALL);

  double dx[3];
  double denom = (double)numpoints_;
  for (int i = 0; i < 3; i++) {
    dx[i] = (p2_[i] - p1_[i]) / denom;
  }
  for (int i = 0; i < 3; i++) {
    point(i) = p1_[i] + 0.5 * dx[i] + dx[i] * (float)pointId;
  }
}
#endif

void
ActSimpleWriteToFile(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta)
{
  if (!actMeta.has_output_file_)
    return;
  std::string filename = actMeta.output_filenames_[actBulk.localTurbineId_];
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  auto vel = helper.get_local_view(actBulk.velocity_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto relVel = helper.get_local_view(actBulk.relativeVelocity_);
  auto density = helper.get_local_view(actBulk.density_);
  const int offset = actBulk.turbIdOffset_.h_view(actBulk.localTurbineId_);

  if (actBulk.localTurbineId_ == NaluEnv::self().parallel_rank()) {
    std::ofstream outFile;
    // ThrowErrorIf(NaluEnv::self().parallel_rank()!=0);

    outFile.open(filename, std::ios_base::app);
    const int stop =
      offset + actMeta.numPointsTurbine_.h_view(actBulk.localTurbineId_);

    for (int index = offset; index < stop; ++index) {
      const int i = index - offset;
      // write cached stuff from earlier computations
      outFile << actBulk.output_cache_[i];
      outFile << vel(index, 0) << ", " << vel(index, 1) << ", " << vel(index, 2)
              << ", ";
      outFile << relVel(index, 0) << ", " << relVel(index, 1) << ", "
              << relVel(index, 2) << ", ";
      outFile << force(index, 0) << ", " << force(index, 1) << ", "
              << force(index, 2) << ", ";
      outFile << density(index) << std::endl;
      actBulk.output_cache_[i].clear();
    }
    outFile.close();
  }
}

ActSimpleAssignVel::ActSimpleAssignVel(ActuatorBulkSimple& actBulk)
  : velocity_(helper_.get_local_view(actBulk.velocity_)),
    density_(helper_.get_local_view(actBulk.density_)),
    points_(helper_.get_local_view(actBulk.pointCentroid_)),
    offset_(helper_.get_local_view(actBulk.turbIdOffset_)),
    debug_output_(actBulk.debug_output_),
    turbId_(actBulk.localTurbineId_)
{
}

void
ActSimpleAssignVel::operator()(int index) const
{

  const int pointId = index - offset_(turbId_);
  auto vel = Kokkos::subview(velocity_, index, Kokkos::ALL);
  auto rho = Kokkos::subview(density_, index);

  // Use this to double check the velocities and point positions
  auto point = Kokkos::subview(points_, index, Kokkos::ALL);
  if (debug_output_)
    NaluEnv::self().naluOutput()
      << "Blade " << turbId_ // LCCOUT
      << " pointId: " << pointId << std::scientific << std::setprecision(5)
      << " point: " << point(0) << " " << point(1) << " " << point(2) << " "
      << " vel: " << vel(0) << " " << vel(1) << " " << vel(2) << " "
      << " rho: " << *rho.data() << std::endl;
  // Do nothing otherwise
}

void
ActSimpleComputeRelativeVelocity(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta)
{
  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  helper.touch_dual_view(actBulk.alpha_);
  helper.touch_dual_view(actBulk.relativeVelocity_);
  auto velocity = helper.get_local_view(actBulk.velocity_);
  auto relVelocity = helper.get_local_view(actBulk.relativeVelocity_);
  auto alpha = helper.get_local_view(actBulk.alpha_);
  auto offset = helper.get_local_view(actBulk.turbIdOffset_);

  const int turbId = actBulk.localTurbineId_;

  Kokkos::deep_copy(alpha, 0);
  Kokkos::deep_copy(relVelocity, 0);

  Kokkos::parallel_for(
    "compute relative velocities", actBulk.local_range_policy(),
    ACTUATOR_LAMBDA(int index) {
      auto twistTable = Kokkos::subview(
        helper.get_local_view(actMeta.twistTableDv_), turbId, Kokkos::ALL);
      auto p1ZeroAlphaDir = Kokkos::subview(
        helper.get_local_view(actMeta.p1ZeroAlphaDir_), turbId, Kokkos::ALL);
      auto chordNormalDir = Kokkos::subview(
        helper.get_local_view(actMeta.chordNormalDir_), turbId, Kokkos::ALL);
      auto spanDir = Kokkos::subview(
        helper.get_local_view(actMeta.spanDir_), turbId, Kokkos::ALL);
      const int i = index - offset(turbId);

      auto vel = Kokkos::subview(velocity, index, Kokkos::ALL);
      auto relVel = Kokkos::subview(relVelocity, index, Kokkos::ALL);
      double twist = twistTable(i);

      double ws[3] = {vel(0), vel(1), vel(2)}; // Total wind speed

      // Calculate the angle of attack (AOA) and 2d velocity
      AirfoilTheory2D::calculate_alpha(
        ws, p1ZeroAlphaDir.data(), spanDir.data(), chordNormalDir.data(), twist,
        relVel.data(), alpha(index));
    });

  actuator_utils::reduce_view_on_host(alpha);
  actuator_utils::reduce_view_on_host(relVelocity);
}

void
ActSimpleComputeForce(
  ActuatorBulkSimple& actBulk, const ActuatorMetaSimple& actMeta)
{

  ActDualViewHelper<ActuatorFixedMemSpace> helper;
  helper.touch_dual_view(actBulk.actuatorForce_);

  auto density = helper.get_local_view(actBulk.density_);
  auto force = helper.get_local_view(actBulk.actuatorForce_);
  auto offset = helper.get_local_view(actBulk.turbIdOffset_);
  auto alpha = helper.get_local_view(actBulk.alpha_);
  auto relVelocity = helper.get_local_view(actBulk.relativeVelocity_);

  auto aoaPolarTable = helper.get_local_view(actMeta.aoaPolarTableDv_);
  auto clPolarTable = helper.get_local_view(actMeta.clPolarTableDv_);
  auto cdPolarTable = helper.get_local_view(actMeta.cdPolarTableDv_);
  auto elemArea = helper.get_local_view(actMeta.elemAreaDv_);
  auto spanDirection = helper.get_local_view(actMeta.spanDir_);

  const int turbId = actBulk.localTurbineId_;
  const unsigned nPolarTable = actMeta.polarTableSize_.h_view(turbId);

  const int debug_output = actBulk.debug_output_;
  std::vector<std::string>* cache = &actBulk.output_cache_;

  Kokkos::parallel_for(
    "ActSimpleComputeForce", actBulk.local_range_policy(),
    ACTUATOR_LAMBDA(int index) {
      auto pointForce = Kokkos::subview(force, index, Kokkos::ALL);
      const int localId = index - offset(turbId);

      auto ws2d = Kokkos::subview(relVelocity, index, Kokkos::ALL);

      // set up the polar tables
      double* polarPointer =
        Kokkos::subview(aoaPolarTable, turbId, Kokkos::ALL).data();
      double* clPointer =
        Kokkos::subview(clPolarTable, turbId, Kokkos::ALL).data();
      double* cdPointer =
        Kokkos::subview(cdPolarTable, turbId, Kokkos::ALL).data();

      std::vector<double> aoatable(polarPointer, polarPointer + nPolarTable);
      std::vector<double> cltable(clPointer, clPointer + nPolarTable);
      std::vector<double> cdtable(cdPointer, cdPointer + nPolarTable);

      auto spanDir = Kokkos::subview(spanDirection, turbId, Kokkos::ALL);

      // Calculate Cl and Cd
      double cl;
      double cd;
      utils::linear_interp(aoatable, cltable, alpha(index), cl);
      utils::linear_interp(aoatable, cdtable, alpha(index), cd);

      // Magnitude of wind speed
      double ws2Dnorm =
        sqrt(ws2d(0) * ws2d(0) + ws2d(1) * ws2d(1) + ws2d(2) * ws2d(2));

      // Calculate lift and drag forces
      double rho = density(index);
      double area = elemArea(turbId, localId);
      double Q = 0.5 * rho * ws2Dnorm * ws2Dnorm;
      double lift = cl * Q * area;
      double drag = cd * Q * area;

      // Set the directions
      double ws2Ddir[3]; // Direction of drag force
      if (ws2Dnorm > 0.0) {
        ws2Ddir[0] = ws2d(0) / ws2Dnorm;
        ws2Ddir[1] = ws2d(1) / ws2Dnorm;
        ws2Ddir[2] = ws2d(2) / ws2Dnorm;
      } else {
        ws2Ddir[0] = 0.0;
        ws2Ddir[1] = 0.0;
        ws2Ddir[2] = 0.0;
      }
      double liftdir[3]; // Direction of lift force
      if (ws2Dnorm > 0.0) {
        liftdir[0] = ws2Ddir[1] * spanDir(2) - ws2Ddir[2] * spanDir(1);
        liftdir[1] = ws2Ddir[2] * spanDir(0) - ws2Ddir[0] * spanDir(2);
        liftdir[2] = ws2Ddir[0] * spanDir(1) - ws2Ddir[1] * spanDir(0);
      } else {
        liftdir[0] = 0.0;
        liftdir[1] = 0.0;
        liftdir[2] = 0.0;
      }

      // Set the pointForce
      pointForce(0) = -(lift * liftdir[0] + drag * ws2Ddir[0]);
      pointForce(1) = -(lift * liftdir[1] + drag * ws2Ddir[1]);
      pointForce(2) = -(lift * liftdir[2] + drag * ws2Ddir[2]);

      if (debug_output)
        NaluEnv::self().naluOutput()
          << "Blade " << turbId // LCCOUT
          << " pointId: " << localId << std::setprecision(5)
          << " alpha: " << alpha(index) << " ws2D: " << ws2d(0) << " "
          << ws2d(1) << " " << ws2d(2) << " "
          << " Cl, Cd: " << cl << " " << cd << " lift, drag = " << lift << " "
          << drag << std::endl;
      if (actMeta.has_output_file_) {
        std::ostringstream stream;
        stream << localId << ", " << alpha(index) << ", " << cl << ", " << cd
               << ", " << lift << ", " << drag << ", ";
        cache->at(localId) += stream.str();
      }
    });

  actuator_utils::reduce_view_on_host(force);
}

void
AirfoilTheory2D::calculate_alpha(
  double ws[],
  const double zeroalphadir[],
  const double spanDir[],
  const double chodrNormalDir[],
  double twist,
  double ws2D[],
  double& alpha)
{
  // Project WS onto 2D plane defined by zeroalpahdir and chodrNormalDir
  double WSspan = ws[0] * spanDir[0] + ws[1] * spanDir[1] + ws[2] * spanDir[2];
  ws2D[0] = ws[0] - WSspan * spanDir[0];
  ws2D[1] = ws[1] - WSspan * spanDir[1];
  ws2D[2] = ws[2] - WSspan * spanDir[2];

  // Project WS2D onto zeroalphadir and chodrNormalDir
  double WStan = ws2D[0] * zeroalphadir[0] + ws2D[1] * zeroalphadir[1] +
                 ws2D[2] * zeroalphadir[2];

  double WSnormal = ws2D[0] * chodrNormalDir[0] + ws2D[1] * chodrNormalDir[1] +
                    ws2D[2] * chodrNormalDir[2];

  double alphaNoTwist = atan2(WSnormal, WStan) * 180.0 / M_PI;

  alpha = alphaNoTwist + twist;
}

void
ActSimpleComputeThrustInnerLoop::operator()(
  const uint64_t,
  const double*,
  double* sourceTerm,
  const double,
  const double scvIp) const
{

  auto offsets = actBulk_.turbIdOffset_.view_host();

  if (NaluEnv::self().parallel_rank() < actBulk_.num_blades_) {
    int turbId = NaluEnv::self().parallel_rank();
    auto thrust = Kokkos::subview(actBulk_.turbineThrust_, turbId, Kokkos::ALL);

    double forceTerm[3];

    for (int i = 0; i < 3; i++) {
      forceTerm[i] = sourceTerm[i] * scvIp;
      thrust(i) += forceTerm[i];
    }
  }
}

void
ActSimpleSpreadForceWhProjInnerLoop::preloop()
{
  actBulk_.actuatorForce_.sync_host();
}

void
ActSimpleSpreadForceWhProjInnerLoop::operator()(
  const uint64_t pointId,
  const double* nodeCoords,
  double* sourceTerm,
  const double dual_vol,
  const double scvIp) const
{

  auto pointCoords =
    Kokkos::subview(actBulk_.pointCentroid_.view_host(), pointId, Kokkos::ALL);

  auto pointForce =
    Kokkos::subview(actBulk_.actuatorForce_.view_host(), pointId, Kokkos::ALL);

  auto epsilon =
    Kokkos::subview(actBulk_.epsilon_.view_host(), pointId, Kokkos::ALL);

  auto orientation = Kokkos::subview(
    actBulk_.orientationTensor_.view_host(), pointId, Kokkos::ALL);

  double distance[3] = {0, 0, 0};
  double projectedDistance[3] = {0, 0, 0};
  double projectedForce[3] = {0, 0, 0};

  actuator_utils::compute_distance(
    3, nodeCoords, pointCoords.data(), &distance[0]);

  // transform distance from Cartesian to blade coordinate system
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      projectedDistance[i] += distance[j] * orientation(i + j * 3);
    }
  }

  const double gauss = actuator_utils::Gaussian_projection(
    3, &projectedDistance[0], epsilon.data());

  for (int j = 0; j < 3; j++) {
    projectedForce[j] = gauss * pointForce(j);
  }

  for (int j = 0; j < 3; j++) {
    sourceTerm[j] += projectedForce[j] * scvIp / dual_vol;
  }
}

} /* namespace nalu */
} /* namespace sierra */
