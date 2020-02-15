// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <InitialConditions.h>
#include <NaluEnv.h>
#include <Realm.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// InitialCondition - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
  
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- load -----------------------------------------------
//--------------------------------------------------------------------------


/// this is an example of a load() method with polymorphism - the type of
/// the node is determined from some information, then a particular type
/// of object is created and returned to the parent.

InitialCondition * InitialCondition::load(const YAML::Node & node) 
{
   if ( node["constant"] ){
    NaluEnv::self().naluOutputP0() << "Initial Is Type constant " << std::endl;
    ConstantInitialConditionData& constIC = *new ConstantInitialConditionData(*parent());
    node >> constIC;
    return &constIC;
  }
  else  if ( node["user_function"] ){
    NaluEnv::self().naluOutputP0() << "Initial Is Type user-function " << std::endl;
    UserFunctionInitialConditionData& fcnIC = *new UserFunctionInitialConditionData(*parent());
    node >> fcnIC;
    return &fcnIC;
  }
  else
    throw std::runtime_error("parser error InitialConditions::load; unsupported IC type");
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
  return 0;
#endif
}

  Simulation* InitialCondition::root() { return parent()->root(); }
  InitialConditions *InitialCondition::parent() { return &initialConditions_; }

  Simulation* InitialConditions::root() { return parent()->root(); }
  Realm *InitialConditions::parent() { return &realm_; }

InitialConditions* InitialConditions::load(const YAML::Node& node)
{
  InitialCondition tmp_initial_condition(*this);

  if(node["initial_conditions"]) {
    const YAML::Node initial_conditions = node["initial_conditions"];
    for ( size_t j_initial_condition = 0; j_initial_condition < initial_conditions.size(); ++j_initial_condition ) {
      const YAML::Node initial_condition_node = initial_conditions[j_initial_condition];
      InitialCondition* ic = tmp_initial_condition.load(initial_condition_node);
      initialConditionVector_.push_back(ic);
    }
  }
  else {
    throw std::runtime_error("parser error InitialConditions::load");
  }

  return this;
}

} // namespace nalu
} // namespace Sierra
