// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <xfer/Transfers.h>
#include <xfer/Transfer.h>
#include <Simulation.h>
#include <Realms.h>
#include <Realm.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>

#include <stk_mesh/base/BulkData.hpp>

// basic c++
#include <vector>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// Transfers - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Transfers::Transfers(Simulation& sim) : simulation_(sim)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Transfers::~Transfers()
{
  for (size_t ir = 0; ir < transferVector_.size(); ir++)
    delete transferVector_[ir];
}

void
Transfers::load(const YAML::Node& node)
{
  // xfers are optional...
  const YAML::Node transfers = node["transfers"];
  if (transfers) {
    for (size_t itransfer = 0; itransfer < transfers.size(); ++itransfer) {
      const YAML::Node transferNode = transfers[itransfer];
      Transfer* transferInfo = new Transfer(*this);
      transferInfo->load(transferNode);
      transferVector_.push_back(transferInfo);
    }
  }
}

void
Transfers::breadboard()
{
  for (size_t itransfer = 0; itransfer < transferVector_.size(); ++itransfer) {
    transferVector_[itransfer]->breadboard();
  }
}

void
Transfers::initialize()
{
  for (size_t itransfer = 0; itransfer < transferVector_.size(); ++itransfer) {
    transferVector_[itransfer]->initialize_begin();
  }

  for (size_t itransfer = 0; itransfer < transferVector_.size(); ++itransfer) {
    stk::mesh::BulkData& fromBulkData =
      transferVector_[itransfer]->fromRealm_->bulk_data();
    fromBulkData.modification_begin();
    transferVector_[itransfer]->change_ghosting();
    fromBulkData.modification_end();
  }

  for (size_t itransfer = 0; itransfer < transferVector_.size(); ++itransfer) {
    transferVector_[itransfer]->initialize_end();
  }
}

void
Transfers::execute()
{
  for (size_t itransfer = 0; itransfer < transferVector_.size(); ++itransfer) {
    transferVector_[itransfer]->execute();
  }
}

Simulation*
Transfers::root()
{
  return parent()->root();
}
Simulation*
Transfers::parent()
{
  return &simulation_;
}

} // namespace nalu
} // namespace sierra
