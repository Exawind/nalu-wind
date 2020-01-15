// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "overset/UpdateOversetFringeAlgorithmDriver.h"
#include "Realm.h"
#include "overset/OversetManager.h"

#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

UpdateOversetFringeAlgorithmDriver::UpdateOversetFringeAlgorithmDriver(
  Realm& realm)
  : AlgorithmDriver(realm)
{}

UpdateOversetFringeAlgorithmDriver::~UpdateOversetFringeAlgorithmDriver()
{}

void
UpdateOversetFringeAlgorithmDriver::pre_work()
{
  for (auto& f: fields_) {
    realm_.oversetManager_->overset_orphan_node_field_update(
      f->field_, f->sizeRow_, f->sizeCol_);
  }
}

}  // nalu
}  // sierra
