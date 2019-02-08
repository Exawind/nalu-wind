/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * ActuatorDiskFAST.C
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#include "ActuatorDiskFAST.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include <nalu_make_unique.h>

namespace sierra {
namespace nalu {

ActuatorDiskFASTInfo::ActuatorDiskFASTInfo()
  : ActuatorFASTInfo(), bladeSweptPts_(-1){};

// constructor
ActuatorDiskFAST::ActuatorDiskFAST(Realm& realm, const YAML::Node& node)
  : ActuatorFAST(realm, node)
{
  parse_disk_specific(node);
}

void
ActuatorDiskFAST::parse_disk_specific(const YAML::Node& y_node)
{
  int nSwept = 0;
  int nFpts = 0;
  const YAML::Node y_actuator = y_node["actuator"];
  for (int i = 0; i < fi.nTurbinesGlob; i++) {
    const YAML::Node cur_turbine = y_actuator["Turbine" + std::to_string(i)];
    get_if_present(cur_turbine, "num_swept_pts", nSwept);
    useUniformAziSampling_ = nSwept != 0;
    get_required(cur_turbine, "num_force_pts_blade", nFpts);
    numSweptPointMap_.insert(
      std::make_pair(i, std::vector<int>(nFpts, nSwept)));
  }
}

void
ActuatorDiskFAST::update_class_specific()
{
  Actuator::complete_search();
}

void
ActuatorDiskFAST::execute_class_specific(
  const int nDim,
  const stk::mesh::FieldBase* coordinates,
  stk::mesh::FieldBase* actuator_source,
  const stk::mesh::FieldBase* dualNodalVolume)
{

  std::vector<double> ws_pointForce(3);

  // zero out average forces from previous step
  for (auto&& turbine : averageForcesMap_) {
    for (auto&& radPoint : turbine.second) {
      std::fill(radPoint.begin(), radPoint.end(), 0.0);
    }
  }

  // assemble forcing at each radius on the blades
  for (std::size_t i = 0; i < numFastPoints_; i++) {
    auto infoObject =
      dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(i).get());
    int np = static_cast<int>(i);

    if (infoObject->nodeType_ == fast::BLADE) {
      int radialIndex = pointRadiusMap_.at(np);
      std::size_t turbineNum = infoObject->globTurbId_;

      FAST.getForce(ws_pointForce, np, infoObject->globTurbId_);
      std::vector<double>& avgVec =
        averageForcesMap_.at(turbineNum)[radialIndex];
      for (int j = 0; j < 3; j++) {
        avgVec[j] += ws_pointForce[j];
      }
    }
  }

  // average the load at each radius by the number of entries
  for (auto&& turbine : averageForcesMap_) {
    std::vector<std::vector<double>>& avgVec = turbine.second;
    const int np = turbine.first;
    const double numBlades = static_cast<double>(FAST.get_numBlades(np));
    std::vector<int>& numSweptPoints = numSweptPointMap_.at(np);

    for (std::size_t i = 0; i < avgVec.size(); i++) {
      for (std::size_t j = 0; j < 3; j++) {
        avgVec[i][j] /= numBlades * (numSweptPoints[i] + 1);
      }
    }
  }

  // loop over map and assemble source terms
  for (auto&& iterPoint : actuatorPointInfoMap_) {

    // actuator line info object of interest
    auto infoObject =
      dynamic_cast<ActuatorFASTPointInfo*>(iterPoint.second.get());
    int np = static_cast<int>(iterPoint.first);
    if (infoObject == NULL) {
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct "
                               "type.  Should be ActuatorFASTPointInfo.");
    }

    int iTurbGlob = infoObject->globTurbId_;
    std::vector<double> hubPos(3);
    std::vector<double> hubShftVec(3);
    FAST.getHubPos(hubPos, iTurbGlob);
    FAST.getHubShftDir(hubShftVec, iTurbGlob);

    // get the vector of elements
    std::set<stk::mesh::Entity> nodeVec = infoObject->nodeVec_;

    switch (infoObject->nodeType_) {
    case fast::HUB:
    case fast::TOWER:
      FAST.getForce(ws_pointForce, np, iTurbGlob);
      spread_actuator_force_to_node_vec(
        nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]),
        *coordinates, *actuator_source, *dualNodalVolume, infoObject->epsilon_,
        hubPos, hubShftVec, thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::BLADE:
      spread_actuator_force_to_node_vec(
        nDim, nodeVec, averageForcesMap_.at(iTurbGlob)[pointRadiusMap_.at(np)],
        &(infoObject->centroidCoords_[0]), *coordinates, *actuator_source,
        *dualNodalVolume, infoObject->epsilon_, hubPos, hubShftVec,
        thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::ActuatorNodeType_END:
      break;
    default:
      break;
    }
  }
}

Point
ActuatorDiskFAST::get_blade_point_location(int turbNum, int bNum, int radIndex)
{
  std::vector<double> temp(3, 0);
  const int numPntsBlade = FAST.get_numForcePtsBlade(turbNum);
  FAST.getForceNodeCoordinates(
    temp, radIndex + bNum * numPntsBlade + 1, turbNum);
  Point answer = {temp[0], temp[1], temp[2]};
  return answer;
}

void
ActuatorDiskFAST::add_swept_points_to_map()
{
  // Setup storage for the blade averages on this proc
  const int myProcId = NaluEnv::self().parallel_rank();
  const int numTurbGlob = FAST.get_nTurbinesGlob();
  std::map<int, std::vector<double>> towerRadiusMap;
  SweptPointLocator locator;
  Point centroidCoords;

  for (int iTurb = 0; iTurb < numTurbGlob; iTurb++) {
    if (FAST.get_procNo(iTurb) == myProcId) {

      // pull some stuff from fast for this turbine
      const int numPntsBlade = FAST.get_numForcePtsBlade(iTurb);
      const int numBlades = FAST.get_numBlades(iTurb);

      std::vector<int>& mySwept = numSweptPointMap_.at(iTurb);

      const auto actuatorInfo =
        dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_[iTurb].get());
      if (actuatorInfo == NULL) {
        throw std::runtime_error("Object in ActuatorInfo is not the correct "
                                 "type. It should be ActuatorFASTInfo.");
      }
      // set up search params and constants for this fastPoint
      double searchRadius = actuatorInfo->epsilon_.x_ * sqrt(log(1.0 / 0.001));

      // loop over each radial location and insert the correct number of points

      // compute dR for non-uniform azimuthal averaging
      // we will make the arc length between points equal to
      // dR for the non-uniform sampling
      Point r1, r2;
      r1 = get_blade_point_location(iTurb, 0, 0);
      r2 = get_blade_point_location(iTurb, 0, 1);
      double dR = 0;
      for (int d = 0; d < 3; d++) {
        dR += std::pow(r1[d] - r2[d], 2);
      }
      dR = std::sqrt(dR);

      for (int i = numPntsBlade - 1; i >= 0; i--) {
        for (int j = 0; j < numBlades; j++) {
          locator.update_point_location(
            j, get_blade_point_location(iTurb, j, i));
        }
        // get radius and update mySwept
        double radius = locator.get_radius(0);

        if (!useUniformAziSampling_) {
          mySwept[i] = (int)(2.0 * M_PI * radius / numBlades / dR);
        }

        // periodic function has blades points at pi/3, pi, and 5*pi/3
        // this is due to the way the control points are defined
        double theta = M_PI / 3.0;
        for (int b = 0; b < numBlades; b++) {
          const double dtheta = 2.0 * M_PI / (numBlades * (mySwept[i] + 1));
          for (int j = 0; j < mySwept[i]; j++) {
            theta += dtheta;
            centroidCoords = locator(theta);

            // create the bounding point sphere and push back
            std::size_t np = actuatorPointInfoMap_.size();
            stk::search::IdentProc<uint64_t, int> theIdent(np, myProcId);
            boundingSphere theSphere(
              Sphere(centroidCoords, searchRadius), theIdent);
            boundingSphereVec_.push_back(theSphere);

            actuatorPointInfoMap_.insert(std::make_pair(
              np,
              make_unique<ActuatorFASTPointInfo>(
                iTurb, centroidCoords, searchRadius,
                actuatorInfo->epsilon_, // TODO(psakiev)::scale epsilon based on
                                        // the arc length ratio with the tip
                fast::BLADE, i)));
            pointRadiusMap_.insert(std::make_pair(np, i));
          }
          // extra adition to skip over the angle the blade is at
          theta += dtheta;
        }
      }
      // if a turbine was asked to be dumped this will write
      // its points to a file
      ActuatorFAST::dump_turbine_points_to_file(iTurb);
    }
  }
}

void
ActuatorDiskFAST::create_point_info_map_class_specific()
{
  // Setup storage for the blade averages on this proc
  const int myProcId = NaluEnv::self().parallel_rank();
  const int nGTurb = FAST.get_nTurbinesGlob();

  // setup averageForceMap
  std::vector<double> dummyToInsert(3, 0);
  for (int i = 0; i < nGTurb; i++) {
    if (myProcId == FAST.get_procNo(i)) {
      int numPntsBlade = FAST.get_numForcePtsBlade(i);
      averageForcesMap_.insert(std::make_pair(
        i, std::vector<std::vector<double>>(numPntsBlade, dummyToInsert)));
    }
  }

  // Add swept points to the point map
  if (!FAST.isDryRun()) {
    // loop over all the force points
    for (std::size_t i = 0; i < numFastPoints_; i++) {
      ActuatorFASTPointInfo* fastPoint =
        dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(i).get());
      const std::size_t iTurb = fastPoint->globTurbId_;
      const int turbProcId = FAST.get_procNo(iTurb);
      // do something if it's a blade and on this processor
      if (fastPoint->nodeType_ == fast::BLADE && turbProcId == myProcId) {

        int nForcePointsBlade =
          FAST.get_numForcePtsBlade(static_cast<int>(iTurb));

        const auto actuatorInfo =
          dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_[iTurb].get());
        if (actuatorInfo == NULL) {
          throw std::runtime_error("Object in ActuatorInfo is not the correct "
                                   "type. It should be ActuatorFASTInfo.");
        }

        // compute the radius index
        int radIndex = (i - 1) % nForcePointsBlade;
        pointRadiusMap_.insert(std::make_pair(static_cast<int>(i), radIndex));
      }
    }
    add_swept_points_to_map();
  }
}

std::string
ActuatorDiskFAST::get_class_name()
{
  return "ActuatorDiskFAST";
}
//--------------------------------------------------------------------------------------
//  Swept Point Locator
//--------------------------------------------------------------------------------------
SweptPointLocator::SweptPointLocator()
  : bladePoints_(3), controlPoints_(3), controlPointsCurrent_{false}
{
}

void
SweptPointLocator::update_point_location(int i, Point p)
{
  bladePoints_[i] = p;
  controlPointsCurrent_ = false;
}

// Set control points of the Bezier curve so that the blade points
// are on the resulting parametric curve.  This is ensured when the control
// points are 1) on the vector created by the blade point and centroid of blade
// points, 2) and are the blade points are the mid-points of lines connecting
// the control points. See figure 5 in the reference paper for clarification.
void
SweptPointLocator::generate_control_points()
{
  for (int d = 0; d < 3; d++) {
    controlPoints_[2][d] =
      bladePoints_[0][d] + bladePoints_[1][d] - bladePoints_[2][d];
    controlPoints_[1][d] = 2.0 * bladePoints_[1][d] - controlPoints_[2][d];
    controlPoints_[0][d] = 2.0 * bladePoints_[0][d] - controlPoints_[2][d];
  }
  controlPointsCurrent_ = true;
}

int
SweptPointLocator::binomial_coefficient(int N, int R)
{
  int coefficient{1};
  int upperLim = std::max(N - R, R);
  int lowerLim = std::min(N - R, R);
  for (int n = N; n > upperLim; n--) {
    coefficient *= n;
  }
  for (int r = lowerLim; r > 0; r--) {
    coefficient /= r;
  }
  return coefficient;
}

double
SweptPointLocator::periodic_basis(double t)
{
  int binom = binomial_coefficient(order_, order_ / 2);
  double denominator = static_cast<double>((order_ + 1) * binom);
  double eta = std::pow(2.0, order_) / denominator;
  return eta * std::pow(std::cos(0.5 * t), order_);
}

Point
SweptPointLocator::operator()(double t)
{
  Point output = {0, 0, 0};

  if (!controlPointsCurrent_) {
    generate_control_points();
  }

  for (int i = 0; i <= order_; i++) {
    const double offset = i * delta_;
    const double basis = periodic_basis(t - offset);
    for (int k = 0; k < 3; k++) {
      output[k] += controlPoints_[i][k] * basis;
    }
  }

  return output;
}

std::vector<Point>
SweptPointLocator::get_control_points()
{
  generate_control_points();
  return controlPoints_;
}

Point
SweptPointLocator::get_centriod()
{
  Point centroid = {0.0, 0.0, 0.0};

  for (int i = 0; i < 3; i++) {
    centroid[0] += bladePoints_[i][0];
    centroid[1] += bladePoints_[i][1];
    centroid[2] += bladePoints_[i][2];
  }

  centroid[0] /= 3.0;
  centroid[1] /= 3.0;
  centroid[2] /= 3.0;

  return centroid;
}

double
SweptPointLocator::get_radius(int pntNum)
{
  if (!controlPointsCurrent_) {
    generate_control_points();
  }
  double distance{0.0};
  Point centroid = get_centriod();

  for (int i = 0; i < 3; i++) {
    distance += std::pow(bladePoints_[pntNum][i] - centroid[i], 2.0);
  }

  return std::sqrt(distance);
}

} // namespace nalu
} // namespace sierra
