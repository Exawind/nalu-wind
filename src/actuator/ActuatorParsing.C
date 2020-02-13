// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorParsing.h>
#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorInfo.h>
#include <stdexcept>

namespace sierra {
namespace nalu {

void actuator_line_FAST_parse(const YAML::Node& y_node, ActuatorMeta& actMeta){
#ifndef NALU_USES_OPENFAST
  throw std::runtime_error("Compile with OpenFAST to use ActLineFAST");
#endif

}

void actuator_disk_FAST_parse(const YAML::Node& y_node, ActuatorMeta& actMeta){
#ifndef NALU_USES_OPENFAST
  throw std::runtime_error("Compile with OpenFAST to use ActDiskFAST");
#endif

}

/*! \brief Parse parameters to construct meta data for actuators
 *  Parse parameters and construct meta data for actuators.
 *  Intent is to divorce object creation/memory allocation from parsing
 *  to facilitate device compatibility.
 *
 *  This also has the added benefit of increasing unittest-ability.
 *
 *  General parameters that apply to all actuators should be parsed here.
 *  More specific actuator methods (i.e. LineFAST, DiskFAST) should implement
 *  another parse function that takes one YAML::Node and one ActuatorMeta object
 *  as inputs and returns a more specialized ActuatorMeta object.
 */
ActuatorMeta actuator_parse(const YAML::Node& y_node)
{
  const YAML::Node y_actuator = y_node["actuator"];
  if (y_actuator)
  {
    int nTurbines = 0;
    std::string actuatorTypeName;
    get_required(y_actuator, "n_turbines_glob", nTurbines);
    get_required(y_actuator, "type", actuatorTypeName);
    ActuatorMeta actMeta(nTurbines, ActuatorTypeMap[actuatorTypeName]);
    // search specifications
    std::string searchMethodName = "na";
    get_if_present(y_actuator, "search_method", searchMethodName,
      searchMethodName);
    // determine search method for this pair
    if (searchMethodName == "boost_rtree")
    {
      actMeta.searchMethod_ = stk::search::BOOST_RTREE;
      NaluEnv::self().naluOutputP0() << "Warning: search method 'boost_rtree'"
          << " is being deprecated, please switch to 'stk_kdtree'" << std::endl;
    } else if (searchMethodName == "stk_kdtree")
      actMeta.searchMethod_ = stk::search::KDTREE;
    else
      NaluEnv::self().naluOutputP0()
          << "Actuator::search method not declared; will use stk_kdtree"
          << std::endl;
    // extract the set of from target names; each spec is homogeneous in this
    // respect
    const YAML::Node searchTargets = y_actuator["search_target_part"];
    if(searchTargets){
      if (searchTargets.Type() == YAML::NodeType::Scalar) {
         actMeta.searchTargetNames_.resize(1);
         actMeta.searchTargetNames_[0] = searchTargets.as<std::string>();
       } else {
         actMeta.searchTargetNames_.resize(searchTargets.size());
         for (size_t i = 0; i < searchTargets.size(); ++i) {
           actMeta.searchTargetNames_[i] = searchTargets[i].as<std::string>();
         }
       }
    }
    else{
      throw std::runtime_error("Actuator:: search_target_part is not declared.");
    }
    return actMeta;
  } else
  {
    return ActuatorMeta(0);
  }
}

} //namespace nalu
} //namespace sierra
