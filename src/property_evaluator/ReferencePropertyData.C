// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <property_evaluator/ReferencePropertyData.h>

#include <Enums.h>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ReferencePropertyData::ReferencePropertyData()
  : speciesName_("na"),
    mw_(0.0),
    massFraction_(0.0),
    stoichiometry_(0.0),
    primaryMassFraction_(0.0),
    secondaryMassFraction_(0.0)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ReferencePropertyData::~ReferencePropertyData()
{
  // nothing
}

} // namespace nalu
} // namespace sierra
