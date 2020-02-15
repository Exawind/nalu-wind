// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef Transfers_h
#define Transfers_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra{
namespace nalu{

class Simulation;
class Transfer;

class Transfers {
public:
  Transfers(Simulation& sim);
  ~Transfers();

  void load(const YAML::Node & node);
  void breadboard();
  void initialize();
  void execute(); // general method to execute all xfers (as apposed to Realm)
  Simulation *root();
  Simulation *parent();

  Simulation &simulation_;
  std::vector<Transfer *> transferVector_;
};

} // namespace nalu
} // namespace Sierra

#endif
