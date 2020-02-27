// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef Realms_h
#define Realms_h

#include <Enums.h>
#include <Realm.h>

#include <map>
#include <string>
#include <algorithm>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra{
namespace nalu{

typedef std::vector<Realm *> RealmVector;

class Simulation;

class Realms {

public:
  Realms(Simulation& sim) : simulation_(sim) {}
  ~Realms();

  void load(const YAML::Node & node) ;
  void breadboard();
  void initialize();
  Simulation *root();
  Simulation *parent();
  size_t size() {return realmVector_.size();}

  // find realm with operator
  struct IsString {
    IsString(std::string& str) : str_(str) {}
    std::string& str_;
    bool operator()(Realm *realm) { return realm->name_ == str_; }
  };

  Realm *find_realm(std::string realm_name)
  {
    RealmVector::iterator realm_iter 
      = std::find_if(realmVector_.begin(), realmVector_.end(), IsString(realm_name));
    if (realm_iter != realmVector_.end())
      return *realm_iter;
    else
      return 0;
  }

  Simulation &simulation_;
  RealmVector realmVector_;
};

} // namespace nalu
} // namespace Sierra

#endif
