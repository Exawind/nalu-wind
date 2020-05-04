// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <actuator/ActuatorLineSimple.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>
#include <nalu_make_unique.h>
#include "utils/LinearInterpolation.h"

// master elements
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// stk_search
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>

// basic c++
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>

namespace sierra {
namespace nalu {

// constructor
ActuatorLineSimple::ActuatorLineSimple(Realm& realm, const YAML::Node& node)
  : ActuatorSimple(realm, node)
{
}

void
ActuatorLineSimple::update_class_specific()
{
  ActuatorSimple::update();
}

void
ActuatorLineSimple::create_point_info_map_class_specific()
{
}

void
ActuatorLineSimple::execute_class_specific(
  const int nDim,
  const stk::mesh::FieldBase* coordinates,
  stk::mesh::FieldBase* actuator_source,
  const stk::mesh::FieldBase* dual_nodal_volume)
{

  std::vector<double> ws_pointForce(nDim);
  // loop over map and assemble source terms
  for (auto&& iterPoint : actuatorPointInfoMap_) {

    // actuator line info object of interest
    auto infoObject =
      dynamic_cast<ActuatorSimplePointInfo*>(iterPoint.second.get());
    int np = static_cast<int>(iterPoint.first);
    if (infoObject == NULL) {
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct "
                               "type.  Should be ActuatorSimplePointInfo.");
    }

    // Retrieve stuff from infoObject
    size_t bladeId = infoObject->bladeId_;
    auto bladeInfo =
      dynamic_cast<ActuatorSimpleInfo*>(actuatorInfo_.at(bladeId).get());

    // get the vector of elements
    std::set<stk::mesh::Entity> nodeVec = infoObject->nodeVec_;

    // Get the twist at that station
    size_t iNode = infoObject->forcePntIndex_;

    Coordinates windSpeed = infoObject->windSpeed_;
    double      rho       = infoObject->gasDensity_;
    // NaluEnv::self().naluOutputP0()
    //   << "Blade: " <<bladeId<<" Node: "<<iNode<<" "
    //   << infoObject->centroidCoords_[0] << " "
    //   << infoObject->centroidCoords_[1] << " "
    //   << infoObject->centroidCoords_[2] << " | "
    //   << windSpeed.x_ << " "
    //   << windSpeed.y_ << " "
    //   << windSpeed.z_ << " "
    //   << std::endl;  // LCCOUT

    // Calculate alpha
    double      twist          = bladeInfo->twist_table_[iNode];
    double      area           = bladeInfo->elem_area_[iNode];
    Coordinates zeroalphadir   = bladeInfo->p1zeroalphadir_;
    Coordinates chordnormaldir = bladeInfo->chordnormaldir_;
    Coordinates spandir        = bladeInfo->spandir_;
    double      alpha          = 0.0;

    Coordinates ws2D;
    calculate_alpha(windSpeed, zeroalphadir, spandir, chordnormaldir,
		    twist, alpha, ws2D);

    // -- Calculate ws_pointForce --
    // Get CL, CD coefficients
    double cl=0.0;
    double cd=0.0;
    calculate_cl_cd(alpha, 
		    bladeInfo->aoa_polartable_,
		    bladeInfo->cl_polartable_,
		    bladeInfo->cd_polartable_,
		    cl, cd);

    // Magnitude of wind speed
    double ws2Dnorm = sqrt(ws2D.x_*ws2D.x_ + 
			   ws2D.y_*ws2D.y_ +
			   ws2D.z_*ws2D.z_);

    double Q    = 0.5*rho*ws2Dnorm*ws2Dnorm;
    double lift = cl*Q*area;
    double drag = cd*Q*area;

    BladeTotalLift[bladeId] += lift;    
    BladeTotalDrag[bladeId] += drag;    
    BladeAvgAlpha[bladeId]  += alpha;    
    BladeAvgWS2D[bladeId][0] += ws2D.x_;    
    BladeAvgWS2D[bladeId][1] += ws2D.y_;    
    BladeAvgWS2D[bladeId][2] += ws2D.z_;    

    // Set the directions
    Coordinates ws2Ddir;  // Direction of drag force
    if (ws2Dnorm > 0.0) {
      ws2Ddir.x_ = ws2D.x_/ws2Dnorm;
      ws2Ddir.y_ = ws2D.y_/ws2Dnorm;
      ws2Ddir.z_ = ws2D.z_/ws2Dnorm;
    } else {
      ws2Ddir.x_ = 0.0; 
      ws2Ddir.y_ = 0.0; 
      ws2Ddir.z_ = 0.0; 
    }

    Coordinates liftdir;  // Direction of lift force
    if (ws2Dnorm > 0.0) {
      liftdir.x_ = ws2Ddir.y_*spandir.z_ - ws2Ddir.z_*spandir.y_; 
      liftdir.y_ = ws2Ddir.z_*spandir.x_ - ws2Ddir.x_*spandir.z_; 
      liftdir.z_ = ws2Ddir.x_*spandir.y_ - ws2Ddir.y_*spandir.x_; 
    } else {
      liftdir.x_ = 0.0; 
      liftdir.y_ = 0.0; 
      liftdir.z_ = 0.0; 
    }

    if (debug_output_)
      NaluEnv::self().naluOutputP0()
	<< "Blade: " <<bladeId<<" Node: "<<iNode<<" Alpha, Cl, Cd: "
	<<alpha<<" "<<cl<<" "<<cd
	<<" lift, drag = "<<lift<<" "<<drag
	<<std::endl; //LCCOUT
    
    // set ws_pointForce
    ws_pointForce[0] = -(lift*liftdir.x_ + drag*ws2Ddir.x_);
    ws_pointForce[1] = -(lift*liftdir.y_ + drag*ws2Ddir.y_);
    if (nDim>2) 
    ws_pointForce[2] = -(lift*liftdir.z_ + drag*ws2Ddir.z_);

    // Declare the orientation matrix
    // The ordering of this matrix is: xx, xy, xz, yx, yy, yz, zx, zy, zz
    // The default value is a matrix which causes no rotation 
    // This rotation takes into account the fact that the axes, x and y are
    // inverted after the rotation is done inside the 
    // spread_actuator_force_to_node_vec function.
    std::vector<double> orientation_tensor
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    std::vector<double> hubPos(3, 0.0);
    std::vector<double> hubShftVec(3, 0.0);

    // Call the function to spread the node      
    spread_actuator_force_to_node_vec(
      nDim, nodeVec, ws_pointForce, 
      orientation_tensor, // The tensor with the airfoil orientation
      &(infoObject->centroidCoords_[0]),
      *coordinates, *actuator_source, *dual_nodal_volume,
      infoObject->epsilon_, hubPos, hubShftVec, thrust[bladeId],
      torque[bladeId]);

  }
}

std::string
ActuatorLineSimple::get_class_name()
{
  return "ActuatorLineSimple";
}

// Calculates the angle of attack alpha
void 
ActuatorLineSimple::calculate_alpha(
    Coordinates ws, 
    Coordinates zeroalphadir, 
    Coordinates spandir,
    Coordinates chordnormaldir, 
    double twist,
    double &alpha,
    Coordinates &ws2D)
{
  // Project WS onto 2D plane defined by zeroalpahdir and chordnormaldir
  double WSspan = ws.x_*spandir.x_ + ws.y_*spandir.y_ + ws.z_*spandir.z_;
  ws2D.x_ = ws.x_ - WSspan*spandir.x_;
  ws2D.y_ = ws.y_ - WSspan*spandir.y_;
  ws2D.z_ = ws.z_ - WSspan*spandir.z_;

  // Project WS2D onto zeroalphadir and chordnormaldir
  double WStan = 
    ws2D.x_*zeroalphadir.x_ + 
    ws2D.y_*zeroalphadir.y_ +  
    ws2D.z_*zeroalphadir.z_ ;
  
  double WSnormal = 
    ws2D.x_*chordnormaldir.x_ + 
    ws2D.y_*chordnormaldir.y_ + 
    ws2D.z_*chordnormaldir.z_ ;
  

  double alphaNoTwist = atan2(WSnormal, WStan)*180.0/M_PI;
  alpha = alphaNoTwist + twist;  
} // End calculate_alpha

void 
ActuatorLineSimple::calculate_cl_cd(
    double alpha,
    std::vector<double> aoatable,
    std::vector<double> cltable,
    std::vector<double> cdtable,
    double &cl,
    double &cd)
{
  // Get cl and cd from the tables
  utils::linear_interp(aoatable, cltable, alpha, cl);
  utils::linear_interp(aoatable, cdtable, alpha, cd);

  // Do another other processing needed on cl/cd
  // [..nothing at this time..]
}



} // namespace nalu
} // namespace sierra
