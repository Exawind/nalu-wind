/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <Realm.h>
#include <MaterialProperty.h>
#include <MaterialPropertys.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// MaterialProperty - material property
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MaterialProperty::MaterialProperty(
  MaterialPropertys &matPropertys)
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
MaterialProperty::load(const YAML::Node & /* node */) 
{
  // nothing...
}


} // namespace nalu
} // namespace Sierra
