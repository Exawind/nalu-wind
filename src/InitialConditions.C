/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


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

} // namespace nalu
} // namespace Sierra
