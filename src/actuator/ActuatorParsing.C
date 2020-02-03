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

namespace sierra {
namespace nalu {

namespace actuator{

ActuatorMeta actuator_parse(const YAML::Node& y_node){
  const YAML::Node y_actuator = y_node["actuator"];
  if(y_actuator){
    int nTurbines = 0;
    get_required(y_actuator, "n_turbines_glob", nTurbines);
    ActuatorMeta actMeta(nTurbines);
    if(y_actuator["type"]){
      const std::string ActuatorTypeName = y_actuator["type"].as<std::string>();
      switch ( ActuatorTypeMap[ActuatorTypeName] ) {
        case ActuatorType::ActLineFAST : {
          actuator_line_FAST_parse(y_actuator, actMeta);
          break;
          throw std::runtime_error("look_ahead_and_create::error: Requested actuator type: " + ActuatorTypeName + ", but was not enabled at compile time");
        // Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
          break;
#endif
        }
        case ActuatorType::ActDiskFAST : {
          //actuator_disk_FAST_parse(y_actuator, actMeta);
          break;
          throw std::runtime_error("look_ahead_and_create::error: Requested actuator type: " + ActuatorTypeName + ", but was not enabled at compile time");
    // Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
          break;
#endif
        }
        default : {
          throw std::runtime_error("look_ahead_and_create::error: unrecognized actuator type: " + ActuatorTypeName);
    // Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
          break;
#endif
        }
      }
    }
    else {
      throw std::runtime_error("look_ahead_and_create::error: No 'type' specified in actuator");
    }

    return actMeta;
  }
  else{
    return ActuatorMeta(0);
  }
}

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

} //namespace actuator
} //namespace nalu
} //namespace sierra
