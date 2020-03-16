// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

/*
 * ActuatorDiskFAST.C
 *
 *  Created on: Oct 16, 2018
 *      Author: psakiev
 */

#include <actuator/ActuatorDiskFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <nalu_make_unique.h>

namespace sierra {
namespace nalu {

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

    // Declare the unit vector orientation matrix
    // This is meant to not rotate the coordinate system
    // The ordering of this matrix is: xx, xy, xz, yx, yy, yz, zx, zy, zz
    const std::vector<double> orientation_tensor
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    switch (infoObject->nodeType_) {
    case fast::HUB:
    case fast::TOWER:
      FAST.getForce(ws_pointForce, np, iTurbGlob);
      spread_actuator_force_to_node_vec(
        nDim, nodeVec, ws_pointForce, 
        // orientation tensor (does not change anything in the actuator disk)
        orientation_tensor,
        &(infoObject->centroidCoords_[0]),
        *coordinates, *actuator_source, *dualNodalVolume, infoObject->epsilon_,
        hubPos, hubShftVec, thrust[iTurbGlob], torque[iTurbGlob]);
      break;
    case fast::BLADE:
      spread_actuator_force_to_node_vec(
        nDim, nodeVec, averageForcesMap_.at(iTurbGlob)[pointRadiusMap_.at(np)],
        // orientation tensor (does not change anything in the actuator disk)
        orientation_tensor,
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
  actuator_utils::SweptPointLocator locator;
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
                                             actuatorInfo->epsilon_,
                                             actuatorInfo->epsilon_,
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


} // namespace nalu
} // namespace sierra
