// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <actuator/ActuatorLineFAST.h>
#include <FieldTypeDef.h>
#include <NaluParsing.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Simulation.h>
#include <nalu_make_unique.h>

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
ActuatorLineFAST::ActuatorLineFAST(Realm& realm, const YAML::Node& node)
  : ActuatorFAST(realm, node)
{
}

void
ActuatorLineFAST::update_class_specific()
{
  ActuatorFAST::update();
}

void
ActuatorLineFAST::create_point_info_map_class_specific()
{
}

void
ActuatorLineFAST::execute_class_specific(
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
      dynamic_cast<ActuatorFASTPointInfo*>(iterPoint.second.get());
    int np = static_cast<int>(iterPoint.first);
    if (infoObject == NULL) {
      throw std::runtime_error("Object in ActuatorPointInfo is not the correct "
                               "type.  Should be ActuatorFASTPointInfo.");
    }

    // Get the force from FAST
    FAST.getForce(ws_pointForce, np, infoObject->globTurbId_);

    std::vector<double> hubPos(3);
    std::vector<double> hubShftVec(3);
    int iTurbGlob = infoObject->globTurbId_;
    FAST.getHubPos(hubPos, iTurbGlob);
    FAST.getHubShftDir(hubShftVec, iTurbGlob);

    // get the vector of elements
    std::set<stk::mesh::Entity> nodeVec = infoObject->nodeVec_;

    // Declare the orientation matrix
    // The ordering of this matrix is: xx, xy, xz, yx, yy, yz, zx, zy, zz
    // The default value is a matrix which causes no rotation 
    // This rotation takes into account the fact that the axes, x and y are
    // inverted after the rotation is done inside the 
    // spread_actuator_force_to_node_vec function.
    std::vector<double> orientation_tensor
        {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

    switch (infoObject->nodeType_) {

      case fast::HUB:

        break;

      case fast::BLADE:

        // Obtain the orientation matrix of the coordinate system
        // This rotation matrix will transform the standard x, y, z coordinate 
        //   system to a coordinate system at the blade section reference frame
        //   that is thicknes, chord, spanwise
        FAST.getForceNodeOrientation(orientation_tensor, np, 
          infoObject->globTurbId_);
    
        break;

      case fast::TOWER:
  
        break;

      case fast::ActuatorNodeType_END:

        break;
    }

    // Call the function to spread the node      
    spread_actuator_force_to_node_vec(
      nDim, nodeVec, ws_pointForce, 
      orientation_tensor, // The tensor with the airfoil orientation
      &(infoObject->centroidCoords_[0]),
      *coordinates, *actuator_source, *dual_nodal_volume,
      infoObject->epsilon_, hubPos, hubShftVec, thrust[iTurbGlob],
      torque[iTurbGlob]);
    
  }
}

std::string
ActuatorLineFAST::get_class_name()
{
  return "ActuatorLineFAST";
}

} // namespace nalu
} // namespace sierra
