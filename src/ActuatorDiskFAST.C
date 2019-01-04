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

#ifdef NALU_USES_OPENFAST
#include "ActuatorDiskFAST.h"
#include <cmath>
#include <algorithm>
#include <functional>

namespace sierra{
namespace nalu{

static bool abs_compare(double a, double b);

ActuatorDiskFASTInfo::ActuatorDiskFASTInfo():
    ActuatorFASTInfo(),
    bladeSweptPts_(-1){};

// constructor
ActuatorDiskFAST::ActuatorDiskFAST(
  Realm &realm,
  const YAML::Node &node)
  : ActuatorFAST(realm, node)
{
  parse_disk_specific(node);
  }

void ActuatorDiskFAST::parse_disk_specific( const YAML::Node& y_node){
  int sweptPoints = 0;
  const YAML::Node y_actuator = y_node["actuator"];
  for(int i = 0; i< fi.nTurbinesGlob; i++){
    const YAML::Node cur_turbine = y_actuator["Turbine"+std::to_string(i)];
    get_required(cur_turbine, "num_swept_pts", sweptPoints);
    sweptPointMap_.emplace(i,sweptPoints);
  }
}

void ActuatorDiskFAST::update_class_specific() {
  Actuator::complete_search();
}

void ActuatorDiskFAST::execute_class_specific(
  const int nDim,
  const stk::mesh::FieldBase * coordinates,
  stk::mesh::FieldBase * actuator_source,
  const stk::mesh::FieldBase * dualNodalVolume
  ){

  std::vector<double> ws_pointForce(3);

  // zero out average forces from previous step
  for(auto&& turbine : loadAverageMap_){
    for(auto&& radPoint : turbine.second){
      std::fill(radPoint.begin(),radPoint.end(),0.0);
    }
  }

  // assemble forcing at each radius on the blades
  for(std::size_t i = 0; i < numFastPoints_; i ++){
    auto infoObject = dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(i).get());
    int np = static_cast<int>(i);

    if(infoObject->nodeType_==fast::BLADE){
      int radialIndex = pointRadiusMap_.at(np);
      std::size_t turbineNum = infoObject->globTurbId_;

      FAST.getForce(ws_pointForce, np, infoObject->globTurbId_);
      std::vector<double>& avgVec = loadAverageMap_.at(turbineNum)[radialIndex];
      for (int j=0; j<3; j++){
        avgVec[j] += ws_pointForce[j];
      }
    }
  }

  // average the load at each radius by the number of entries
  for (auto&& turbine : loadAverageMap_){
    std::vector<std::vector<double>>& avgVec = turbine.second;
    const int np = turbine.first;
    const double numBlades = static_cast<double>(FAST.get_numBlades(np));
    const int numSweptPoints = sweptPointMap_.at(np);

    for (std::size_t i=0; i<avgVec.size(); i++){
      for(std::size_t j=0; j<3; j++){
       avgVec[i][j]/=numBlades*(numSweptPoints+1);
      }
    }
  }

  // loop over map and assemble source terms
  for (auto&& iterPoint : actuatorPointInfoMap_) {

    // actuator line info object of interest
    auto infoObject = dynamic_cast<ActuatorFASTPointInfo*>(iterPoint.second.get());
    int np = static_cast<int>(iterPoint.first);
    if( infoObject==NULL){
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct type.  Should be ActuatorFASTPointInfo.");
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
        nDim,
        nodeVec,
        ws_pointForce,
        &(infoObject->centroidCoords_[0]),
        *coordinates,
        *actuator_source,
        *dualNodalVolume,
        infoObject->epsilon_,
        hubPos,
        hubShftVec,
        thrust[iTurbGlob],
        torque[iTurbGlob]
               );
      break;
    case fast::BLADE:
      spread_actuator_force_to_node_vec(
        nDim,
        nodeVec,
        loadAverageMap_.at(iTurbGlob)[pointRadiusMap_.at(np)],
        &(infoObject->centroidCoords_[0]),
        *coordinates,
        *actuator_source,
        *dualNodalVolume,
        infoObject->epsilon_,
        hubPos,
        hubShftVec,
        thrust[iTurbGlob],
        torque[iTurbGlob]
               );
      break;
    case fast::ActuatorNodeType_END:
      break;
    default:
      break;
    }
 }
}

void ActuatorDiskFAST::create_point_info_map_class_specific(){
  // Setup storage for the blade averages on this proc
  const int myProcId = NaluEnv::self().parallel_rank();
  const int numTurbGlob = FAST.get_nTurbinesGlob();
  std::size_t np = numFastPoints_;
  std::map<int, std::vector<double>> towerRadiusMap;
  for(int i =0; i<numTurbGlob; i++){
    if(FAST.get_procNo(i)==myProcId){
      const int numPntsBlade = FAST.get_numForcePtsBlade(i);
      const int totalNumForcePnts = FAST.get_numForcePts(i);
      std::vector<double> temp ={0,0,0};
      loadAverageMap_.emplace(i,std::vector<std::vector<double>>(numPntsBlade,temp));

      //create radius map
      std::vector<double> radVec;
      std::vector<double> pointLoc(3);
      std::vector<double> hubPos(3);
      std::vector<double> dirMatch(3), dirTest(3);
      FAST.getHubPos(hubPos, i);
      bool directionSet = false;

      //TODO(psakiev)::refactor to use setup of num blades and openfast data storage
      for(int j =0; j<totalNumForcePnts; j++){
        if(FAST.getForceNodeType(i,j)==fast::BLADE){
          double radTemp=0;
          FAST.getForceNodeCoordinates(pointLoc, j, i);
          // save direction of first point (hub to point)
          if(!directionSet){
            dirMatch = NormalizedDirection(pointLoc, hubPos);
            directionSet = true;
          }

          // select points with reference to this direction
          dirTest = NormalizedDirection(pointLoc, hubPos);

          std::transform(dirMatch.begin(), dirMatch.end(), dirTest.begin(), dirTest.begin(), std::minus<double>());

          auto maxDiff = std::max_element(dirTest.begin(), dirTest.end(), abs_compare);

          if(*maxDiff < 1.0e-6){
            for(int k=0; k<3; k++){
                radTemp+=std::pow(pointLoc[k]-hubPos[k],2);
              }
              radTemp=std::sqrt(radTemp);
              radVec.push_back(radTemp);
          }
        }
      }
      if(radVec.size() != numPntsBlade){
        throw std::runtime_error("radVec doesn't match force blades. radVec.size() = " +std::to_string(radVec.size()) );
      }

      // sort vector at the end
      std::sort(radVec.begin(), radVec.end());

      // create spatial map for the tower
      towerRadiusMap.emplace(i, radVec);
    }
  }
  // Add swept points to the point map
  if( !FAST.isDryRun() ){
    // loop over all the force points
    for (std::size_t i=0; i<numFastPoints_; i++){
      ActuatorFASTPointInfo* fastPoint = dynamic_cast<ActuatorFASTPointInfo*>(actuatorPointInfoMap_.at(i).get());
      const std::size_t iTurb = fastPoint->globTurbId_;
      const int turbProcId = FAST.get_procNo(iTurb);
      //do something if it's a blade and on this processor
      if(fastPoint->nodeType_==fast::BLADE && turbProcId == myProcId){

        const auto actuatorInfo =
              dynamic_cast<ActuatorFASTInfo*>(actuatorInfo_[iTurb].get());
        if(actuatorInfo == NULL){
           throw std::runtime_error("Object in ActuatorInfo is not the correct type. It should be ActuatorFASTInfo.");
        }

        // set up search params and constants for this fastPoint
        double searchRadius = actuatorInfo->epsilon_.x_*sqrt(log(1.0/0.001));
        std::vector<double> hubPos(3);
        std::vector<double> shaftDirection(3);
        FAST.getHubPos(hubPos, iTurb);
        FAST.getHubShftDir(shaftDirection, iTurb);

        // hard code correct direction for now
        double shaftAngle = -15.0 *std::acos(-1.0)/180.0;
        shaftDirection = {std::cos(shaftAngle),0,-std::sin(shaftAngle)};


        // compute the radius index for the map
        double radius = std::pow(fastPoint->centroidCoords_[0]-hubPos[0],2)+
            std::pow(fastPoint->centroidCoords_[1]-hubPos[1],2)+
            std::pow(fastPoint->centroidCoords_[2]-hubPos[2],2);
        radius=std::sqrt(radius);

        int radIndex = FindClosestIndex(radius,towerRadiusMap[iTurb]);
        if(radIndex>=0){
          pointRadiusMap_.emplace(i,radIndex);
        }
        else{
          throw std::runtime_error("Unable to map radius to index in the actuator disk.");
        }

        const int nSwept = sweptPointMap_.at(iTurb);
        for (int ni = 0; ni < nSwept; ni++){
          // find the spatial location of the swept point
          Point centroidCoords = SweptPointLocation(
            FAST.get_numBlades(iTurb),
            nSwept,
            ni,
            fastPoint->centroidCoords_,
            hubPos,
            shaftDirection);

          // create the bounding point sphere and push back
          stk::search::IdentProc<uint64_t, int> theIdent(np, myProcId);
          boundingSphere theSphere( Sphere(centroidCoords, searchRadius), theIdent);
          boundingSphereVec_.push_back(theSphere);

          actuatorPointInfoMap_.emplace(np, new ActuatorFASTPointInfo
            (
              iTurb, centroidCoords,
              searchRadius, actuatorInfo->epsilon_,
              fast::BLADE
            )
          );
          pointRadiusMap_.emplace(np,radIndex);
          np+=1;

        }
      }
    std::ofstream csvOut;
    csvOut.open("/Users/psakiev/Desktop/turbine"+std::to_string(iTurb)+".csv", std::ofstream::out);
    std::string actOut = ActuatorFAST::write_turbine_points_to_string(iTurb, 10, 8);
    csvOut << actOut;
    csvOut.close();
    }
  }
}

std::string ActuatorDiskFAST::get_class_name(){
  return "ActuatorDiskFAST";
}
//--------------------------------------------------------------------------------------
//  Swept Point Locator
//--------------------------------------------------------------------------------------
SweptPointLocator::SweptPointLocator():
    bladePoints_(3),
    controlPoints_(3),
    controlPointsCurrent_{false}{}

void SweptPointLocator::update_point_location(int i, Point p){
  bladePoints_[i] = p;
  controlPointsCurrent_ = false;
}

// Set control points of the Bezier curve so that the blade points
// are on the resulting parametric curve.  This is ensured when the control
// points are 1) on the vector created by the blade point and centroid of blade points,
// 2) and are the blade points are the mid-points of lines connecting the control points.
// See figure 5 in the reference paper for clarification.
void SweptPointLocator::generate_control_points(){
  for(int d = 0; d<3; d++){
    controlPoints_[2][d] = bladePoints_[0][d]+bladePoints_[1][d]-bladePoints_[2][d];
    controlPoints_[1][d] = 2.0*bladePoints_[1][d]-controlPoints_[2][d];
    controlPoints_[0][d] = 2.0*bladePoints_[0][d]-controlPoints_[2][d];
  }
  controlPointsCurrent_ = true;
}

int SweptPointLocator::binomial_coefficient(int N, int R){
  int coefficient{1};
  int upperLim = std::max(N-R,R);
  int lowerLim = std::min(N-R,R);
  for (int n=N; n>upperLim; n--){
    coefficient*=n;
  }
  for (int r=lowerLim; r>0; r--){
    coefficient/= r;
  }
  return coefficient;
}

double SweptPointLocator::periodic_basis(double t){
  int binom = binomial_coefficient(order_,order_/2);
  double denominator = static_cast<double>((order_+1)*binom);
  double eta = std::pow(2.0,order_)/denominator;
  return eta*std::pow(std::cos(0.5*t),order_);
}

Point SweptPointLocator::operator()(double t){
  Point output={0,0,0};

  if(!controlPointsCurrent_){
    generate_control_points();
  }

  for(int i=0; i<=order_; i++){
    const double offset = i*delta_;
    const double basis = periodic_basis(t-offset);
      for (int k=0; k<3; k++){
        output[k]+=controlPoints_[i][k]*basis;
      }

  }

  return output;
}

std::vector<Point> SweptPointLocator::get_control_points(){
  generate_control_points();
  return controlPoints_;
}

//--------------------------------------------------------------------------------------
//  Free Functions
//--------------------------------------------------------------------------------------
int ComputeNumberOfSweptPoints(double radius, int numBlades, double targetArcLength){
  return static_cast<int>(std::floor(2.0*radius*std::acos(-1.0)/targetArcLength/numBlades));
}

// Determines point locations between actuator lines
Point SweptPointLocation(
  const int nBlades,
  const int nTotalPoints,
  const int nCurrentPoint,
  const Point& pointLoc,
  const std::vector<double>& hubLoc,
  const std::vector<double>& axis){

  Point output;
  // normalize axis
  double magAxis=std::sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
  std::vector<double> nA={axis[0]/magAxis, axis[1]/magAxis, axis[2]/magAxis};

  // compute constants
  double alpha = 2.0*std::acos(-1.0)/nBlades;
  double theta = (alpha * (nCurrentPoint+1))/(nTotalPoints+1);
  std::vector<double> R = {pointLoc[0]-hubLoc[0],pointLoc[1]-hubLoc[1],pointLoc[2]-hubLoc[2]};

  // compute terms (rotate point loc around hub location by theta radians)
  double cosTheta{std::cos(theta)}, sinTheta{std::sin(theta)};
  output[0] =
      R[0]*(cosTheta + nA[0]*nA[0]*(1.0-cosTheta)) +
      R[1]*(nA[0]*nA[1]*(1.0-cosTheta)+nA[2]*sinTheta) +
      R[2]*(nA[0]*nA[2]*(1.0-cosTheta)+nA[1]*sinTheta) +
      hubLoc[0];
  output[1] =
      R[0]*(nA[0]*nA[1]*(1.0-cosTheta)+nA[2]*sinTheta) +
      R[1]*(cosTheta+nA[1]*nA[1]*(1.0-cosTheta)) +
      R[2]*(nA[1]*nA[2]*(1.0-cosTheta)-nA[0]*sinTheta) +
      hubLoc[1];
  output[2] =
      R[0]*(nA[0]*nA[2]*(1.0-cosTheta)-nA[1]*sinTheta) +
      R[1]*(nA[1]*nA[2]*(1.0-cosTheta)+nA[0]*sinTheta) +
      R[2]*(cosTheta+nA[2]*nA[2]*(1.0-cosTheta)) +
      hubLoc[2];

  return output;
}

int FindClosestIndex(const double radius, std::vector<double> vecRad){
  // vecRad must be positive values sorted in ascending order
  // radius must be in the range of values contained by vecRad

  for(auto&& v : vecRad){
    v = std::fabs(radius -v);
  }

  auto it = std::min_element(vecRad.begin(), vecRad.end());
  return std::distance(vecRad.begin(),it);
}

std::vector<double> NormalizedDirection(const std::vector<double>& tip, const std::vector<double>& tail){
  std::vector<double> temp(3);
  double mag {0};
  for(int i=0; i<3; i++){
    temp[i]=tip[i]-tail[i];
    mag+=temp[i]*temp[i];
  }
  mag = std::sqrt(mag);
  if(mag == 0.0){
    return {0,0,0};
  }
  for (int i =0; i<3; i++){
    temp[i]/=mag;
  }
  return temp;
}

static bool abs_compare(double a, double b){
  return (std::fabs(a)<std::fabs(b));
}

}
}
#endif
