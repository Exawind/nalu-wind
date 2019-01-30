/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <ActuatorLineFAST.h>
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

    FAST.getForce(ws_pointForce, np, infoObject->globTurbId_);

    std::vector<double> hubPos(3);
    std::vector<double> hubShftVec(3);
    int iTurbGlob = infoObject->globTurbId_;
    FAST.getHubPos(hubPos, iTurbGlob);
    FAST.getHubShftDir(hubShftVec, iTurbGlob);

    // get the vector of elements
    std::set<stk::mesh::Entity> nodeVec = infoObject->nodeVec_;

    switch (infoObject->nodeType_) {
    case fast::HUB:
    case fast::BLADE:
    case fast::TOWER:
      spread_actuator_force_to_node_vec(
        nDim, nodeVec, ws_pointForce, &(infoObject->centroidCoords_[0]),
        *coordinates, *actuator_source, *dual_nodal_volume,
        infoObject->epsilon_, hubPos, hubShftVec, thrust[iTurbGlob],
        torque[iTurbGlob]);
      break;
    case fast::ActuatorNodeType_END:
      break;
    }
  }
}

std::string
ActuatorLineFAST::get_class_name()
{
  return "ActuatorLineFAST";
}

} // namespace nalu
} // namespace sierra
