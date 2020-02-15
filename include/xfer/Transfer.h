// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef Transfer_h
#define Transfer_h

#include <string>
#include <vector>
#include <utility>
#include <memory>

// stk_transfer related
#include <stk_transfer/TransferBase.hpp>

namespace YAML { class Node; }

// stk
namespace stk {
namespace mesh {
class Part;
typedef std::vector<Part*> PartVector;
}
}

namespace sierra{
namespace nalu{

class Realm;
class Transfers;
class Simulation;

class Transfer
{
public:
  Transfer( Transfers &transfers);
  virtual ~Transfer();

  void load(const YAML::Node & node);

  void breadboard();
  void initialize_begin();
  void change_ghosting();
  void initialize_end();
  void execute();


  Simulation *root();
  Transfers *parent();

  Transfers &transfers_;
  std::shared_ptr<stk::transfer::TransferBase> transfer_;

  bool couplingPhysicsSpecified_;
  bool transferVariablesSpecified_;
  std::string couplingPhysicsName_;

  Realm * fromRealm_;
  Realm * toRealm_;

  // during load
  std::string name_;
  std::string transferType_;
  std::string transferObjective_;
  std::string searchMethodName_;
  double searchTolerance_;
  double searchExpansionFactor_;
  std::pair<std::string, std::string> realmPairName_;
  
  // allow the user to provide a vector "from" and "to" parts; names
  std::vector<std::string> fromPartNameVec_;
  std::vector<std::string> toPartNameVec_;
  // actual parts
  stk::mesh::PartVector fromPartVec_;
  stk::mesh::PartVector toPartVec_;

  // all of the fields
  std::vector<std::pair<std::string, std::string> > transferVariablesPairName_;

  void allocate_stk_transfer();
  void ghost_from_elements();
};


} // namespace nalu
} // namespace Sierra

#endif
