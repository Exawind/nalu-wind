// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <SupplementalAlgorithm.h>
#include <Realm.h>
#include <kernel/Kernel.h>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// Algorithm - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Algorithm::Algorithm(Realm& realm, stk::mesh::Part* part)
  : realm_(realm), partVec_(1, part), fieldManager_(*(realm.fieldManager_.get()))
{
  // nothing to do
}

// alternative; provide full partVec
Algorithm::Algorithm(Realm& realm, const stk::mesh::PartVector& partVec)
  : realm_(realm), partVec_(partVec), fieldManager_(*(realm.fieldManager_.get()))
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Algorithm::~Algorithm()
{
  std::vector<SupplementalAlgorithm*>::iterator ii;
  for (ii = supplementalAlg_.begin(); ii != supplementalAlg_.end(); ++ii)
    delete *ii;

  for (auto* kern : activeKernels_) {
    // Free device copies before cleaning up memory on host
    kern->free_on_device();
    delete kern;
  }
}

} // namespace nalu
} // namespace sierra
