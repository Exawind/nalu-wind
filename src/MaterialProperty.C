// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Realm.h>
#include <MaterialProperty.h>
#include <MaterialPropertys.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// MaterialProperty - material property
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MaterialProperty::MaterialProperty(MaterialPropertys& matPropertys)
  : matPropertys_(matPropertys)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
MaterialProperty::~MaterialProperty()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
MaterialProperty::load(const YAML::Node& /* node */)
{
  // nothing...
}

} // namespace nalu
} // namespace sierra
