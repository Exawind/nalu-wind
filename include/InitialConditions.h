// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef InitialConditions_h
#define InitialConditions_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>

namespace YAML {
  class Node;
}

namespace sierra{
namespace nalu{

class Realm;
class InitialConditions;
class Simulation;

class InitialCondition {
 public:
 InitialCondition(InitialConditions& ics) : initialConditions_(ics), theIcType_(UserDataType_END) {}
  
  virtual ~InitialCondition() {}
  
  InitialCondition * load(const YAML::Node & node) ;
  Simulation *root();
  InitialConditions *parent();
  
  void breadboard()
  {
    // nothing
  }
  
  InitialConditions& initialConditions_;
  
  std::string icName_;
  std::vector<std::string> targetNames_;
  UserDataType theIcType_;
};
 
 typedef std::vector<InitialCondition *> InitialConditionVector;
 
 class InitialConditions {
 public:
 InitialConditions(Realm& realm) : realm_(realm) {}
 
 ~InitialConditions() 
   {
     for ( size_t j_initial_condition = 0; j_initial_condition < initialConditionVector_.size(); ++j_initial_condition ) {
       delete initialConditionVector_[j_initial_condition];
     }
   }
 
   InitialConditions* load(const YAML::Node & node);

 void breadboard()
 {
   for ( size_t j_initial_condition = 0; j_initial_condition < initialConditionVector_.size(); ++j_initial_condition ) {
     initialConditionVector_[j_initial_condition]->breadboard();
   }
 }
 
 Simulation *root();
 Realm *parent();
 
 // ease of access methods to particular initial condition
 size_t size() {return initialConditionVector_.size();}
 InitialCondition *operator[](int i) { return initialConditionVector_[i];}
 
 Realm &realm_;
 InitialConditionVector initialConditionVector_;
}; 
 
} // namespace nalu
} // namespace Sierra

#endif
