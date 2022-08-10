// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <overset/OversetInfo.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// OversetInfo - contains orphan point -> donor elements
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
OversetInfo::OversetInfo(stk::mesh::Entity node, const int nDim)
  : orphanNode_(node),
    owningElement_(),
    bestX_(1.0e16),
    elemIsGhosted_(0),
    meSCS_(NULL)
{
  // resize stuff
  isoParCoords_.resize(nDim);
  nodalCoords_.resize(nDim);
}
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
OversetInfo::~OversetInfo()
{
  // nothing to delete
}

} // namespace nalu
} // namespace sierra
