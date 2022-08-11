// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <AlgorithmDriver.h>

#include <Algorithm.h>
#include <Enums.h>

namespace sierra {
namespace nalu {

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// AlgorithmDriver - Drives algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AlgorithmDriver::AlgorithmDriver(Realm& realm) : realm_(realm)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AlgorithmDriver::~AlgorithmDriver()
{
  std::map<AlgorithmType, Algorithm*>::iterator ii;
  for (ii = algMap_.begin(); ii != algMap_.end(); ++ii) {
    Algorithm* theAlg = ii->second;
    delete theAlg;
  }
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AlgorithmDriver::execute()
{
  pre_work();

  // assemble
  std::map<AlgorithmType, Algorithm*>::iterator it;
  for (it = algMap_.begin(); it != algMap_.end(); ++it) {
    it->second->execute();
  }

  post_work();
}

} // namespace nalu
} // namespace sierra
